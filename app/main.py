import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import pika
import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import AnalysisReport, SessionLocal, get_db, init_db
from app.quant_engine import compute_quantitative_profile

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "analysis_tasks")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AlphaPulse AI",
    description="Asynchronous Multi-Agent Financial Research & Quantitative Analytics Platform",
    version="1.2.0",
    lifespan=lifespan,
)

# Allow upcoming Next.js web client to consume API and SSE streams
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, examples=["AAPL", "THYAO.IS", "BTC-USD"])
    language: Literal["en", "tr"] = Field(default="en", examples=["en", "tr"])

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty.")
        return cleaned


class AnalysisJobResponse(BaseModel):
    job_id: int
    symbol: str
    status: str
    cached: bool = False
    stream_url: str
    message: str


class ReportDetailResponse(BaseModel):
    id: int
    symbol: str
    status: str
    report: str | None = None
    quant_data: dict[str, Any] | None = None
    created_at: str | None = None


def _get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _build_cache_key(symbol: str, language: str) -> str:
    return f"alphapulse:v2:report:{symbol.strip().upper()}:{language.strip().lower()}"


def _publish_to_rabbitmq(payload: dict) -> None:
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),  # Persistent message
        )
    finally:
        connection.close()


@app.get("/health", tags=["System"])
def health_check():
    """Verifies connectivity across API, Redis cache, and RabbitMQ broker."""
    redis_ok = False
    rabbit_ok = False

    try:
        redis_ok = bool(_get_redis_client().ping())
    except Exception:
        redis_ok = False

    try:
        conn = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        rabbit_ok = conn.is_open
        conn.close()
    except Exception:
        rabbit_ok = False

    return {
        "status": "healthy" if (redis_ok and rabbit_ok) else "degraded",
        "services": {
            "api": "up",
            "redis": "up" if redis_ok else "down",
            "rabbitmq": "up" if rabbit_ok else "down",
        },
    }


@app.get("/api/v1/market/{symbol}/quant", tags=["Quantitative Engine"])
def get_instant_quant_profile(symbol: str, period: str = "1y"):
    """
    Computes deterministic technical indicators (RSI, MACD, Bollinger, SMAs),
    risk metrics (Sharpe Ratio, Max Drawdown, Volatility), fundamentals, and
    120-day OHLCV candlestick chart data immediately without invoking LLM agents.
    """
    clean_symbol = symbol.strip().upper()
    try:
        return compute_quantitative_profile(clean_symbol, period=period)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantitative computation failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Analysis"],
)
def request_financial_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Submits a ticker symbol for quantitative + multi-agent financial analysis.
    Serves immediately from Redis if cached; otherwise queues a background job via RabbitMQ.
    """
    cache_key = _build_cache_key(request.symbol, request.language)
    cached_bundle = None

    try:
        raw_cached = _get_redis_client().get(cache_key)
        if raw_cached:
            cached_bundle = json.loads(raw_cached)
    except Exception:
        cached_bundle = None

    # Fast path: serve instantly if already cached in Redis
    if isinstance(cached_bundle, dict) and "report" in cached_bundle:
        record = AnalysisReport(
            symbol=request.symbol,
            status="COMPLETED",
            report=cached_bundle.get("report"),
            quant_data=cached_bundle.get("quant_data"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return AnalysisJobResponse(
            job_id=record.id,
            symbol=record.symbol,
            status=record.status,
            cached=True,
            stream_url=f"/api/v1/reports/{record.id}/stream",
            message="Report and quantitative profile served immediately from Redis cache.",
        )

    # Async path: create PENDING job in PostgreSQL and dispatch to RabbitMQ
    record = AnalysisReport(
        symbol=request.symbol,
        status="PENDING",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        _publish_to_rabbitmq(
            {
                "report_id": record.id,
                "symbol": request.symbol,
                "language": request.language,
            }
        )
    except Exception as exc:
        record.status = "FAILED"
        record.report = f"Failed to enqueue job: {str(exc)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message broker unavailable. Could not enqueue analysis job.",
        ) from exc

    return AnalysisJobResponse(
        job_id=record.id,
        symbol=record.symbol,
        status=record.status,
        cached=False,
        stream_url=f"/api/v1/reports/{record.id}/stream",
        message=f"Analysis queued. Subscribe to /api/v1/reports/{record.id}/stream for live updates.",
    )


@app.get("/api/v1/reports/{report_id}/stream", tags=["Analysis"])
async def stream_report_progress(report_id: int):
    """
    Server-Sent Events (SSE) endpoint streaming real-time agent pipeline progress
    via Redis Pub/Sub until the analysis job completes or fails.
    """
    db = SessionLocal()
    try:
        record = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
        initial_status = record.status
        symbol = record.symbol
    finally:
        db.close()

    async def event_generator():
        if initial_status in ("COMPLETED", "FAILED"):
            done_payload = {
                "report_id": report_id,
                "status": initial_status,
                "step": "FINISHED" if initial_status == "COMPLETED" else "ERROR",
                "message": f"Report for {symbol} is already {initial_status}.",
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            return

        async_redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        pubsub = async_redis.pubsub()
        channel_name = f"alphapulse:stream:{report_id}"
        await pubsub.subscribe(channel_name)

        try:
            init_payload = {
                "report_id": report_id,
                "status": initial_status,
                "step": "STREAM_CONNECTED",
                "message": f"Listening for live agent updates on {symbol}...",
            }
            yield f"data: {json.dumps(init_payload)}\n\n"

            idle_ticks = 0
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("type") == "message":
                    idle_ticks = 0
                    raw_data = msg["data"]
                    yield f"data: {raw_data}\n\n"
                    try:
                        parsed = json.loads(raw_data)
                        if parsed.get("status") in ("COMPLETED", "FAILED"):
                            break
                    except Exception:
                        pass
                else:
                    idle_ticks += 1
                    if idle_ticks % 10 == 0:
                        yield ": keep-alive\n\n"
                    await asyncio.sleep(0.2)
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            await async_redis.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/reports/{report_id}", response_model=ReportDetailResponse, tags=["Analysis"])
def get_report_by_id(report_id: int, db: Session = Depends(get_db)):
    """Retrieves the current status, generated report, and quantitative metrics for a job ID."""
    record = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    return ReportDetailResponse(
        id=record.id,
        symbol=record.symbol,
        status=record.status,
        report=record.report,
        quant_data=record.quant_data,
        created_at=record.created_at.isoformat() if record.created_at else None,
    )


@app.get("/api/v1/reports", response_model=list[ReportDetailResponse], tags=["Analysis"])
def list_recent_reports(limit: int = 10, db: Session = Depends(get_db)):
    """Lists the most recent financial analysis reports."""
    records = (
        db.query(AnalysisReport)
        .order_by(AnalysisReport.id.desc())
        .limit(min(limit, 50))
        .all()
    )
    return [
        ReportDetailResponse(
            id=r.id,
            symbol=r.symbol,
            status=r.status,
            report=r.report,
            quant_data=r.quant_data,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in records
    ]
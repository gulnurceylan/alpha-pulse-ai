import json
import os
from contextlib import asynccontextmanager
from typing import Literal

import pika
import redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import AnalysisReport, get_db, init_db

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
    description="Asynchronous Multi-Agent Financial Research & Institutional Reporting Platform",
    version="1.0.0",
    lifespan=lifespan,
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
    message: str


class ReportDetailResponse(BaseModel):
    id: int
    symbol: str
    status: str
    report: str | None = None
    created_at: str | None = None


def _get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _build_cache_key(symbol: str, language: str) -> str:
    return f"alphapulse:report:{symbol.strip().upper()}:{language.strip().lower()}"


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


@app.post(
    "/api/v1/analyze",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Analysis"],
)
def request_financial_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Submits a ticker symbol for multi-agent financial analysis.
    Serves immediately from Redis if cached; otherwise queues a background job via RabbitMQ.
    """
    cache_key = _build_cache_key(request.symbol, request.language)
    cached_report = None

    try:
        cached_report = _get_redis_client().get(cache_key)
    except Exception:
        cached_report = None

    # Fast path: serve instantly if already cached in Redis
    if cached_report:
        record = AnalysisReport(
            symbol=request.symbol,
            status="COMPLETED",
            report=cached_report,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return AnalysisJobResponse(
            job_id=record.id,
            symbol=record.symbol,
            status=record.status,
            cached=True,
            message="Report served immediately from Redis cache.",
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
        message=f"Analysis queued. Poll GET /api/v1/reports/{record.id} for results.",
    )


@app.get("/api/v1/reports/{report_id}", response_model=ReportDetailResponse, tags=["Analysis"])
def get_report_by_id(report_id: int, db: Session = Depends(get_db)):
    """Retrieves the current status and generated report for a given job ID."""
    record = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    return ReportDetailResponse(
        id=record.id,
        symbol=record.symbol,
        status=record.status,
        report=record.report,
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
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in records
    ]
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pika
import redis
from dotenv import load_dotenv

from app.agents import run_financial_analysis
from app.database import AnalysisReport, SessionLocal, init_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alphapulse.worker")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "analysis_tasks")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


def get_redis_client() -> redis.Redis:
    """Creates a Redis client instance with decoded string responses."""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _build_cache_key(symbol: str, language: str) -> str:
    return f"alphapulse:report:{symbol.strip().upper()}:{language.strip().lower()}"


def _run_analysis_heartbeat_safe(
    connection: pika.BlockingConnection,
    symbol: str,
    language: str,
) -> str:
    """
    Executes the multi-agent analysis in a worker thread while periodically
    servicing RabbitMQ events on the main thread so long-running AI tasks
    never trigger an AMQP heartbeat timeout.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_financial_analysis, symbol, language)
        while not future.done():
            connection.process_data_events(time_limit=1)
            time.sleep(2)
        return future.result()


def process_analysis_job(
    ch: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
) -> None:
    """Processes an incoming financial analysis task from RabbitMQ."""
    db = SessionLocal()
    redis_client = get_redis_client()
    report_record = None

    try:
        payload = json.loads(body.decode("utf-8"))
        report_id = payload.get("report_id")
        symbol = str(payload.get("symbol", "")).strip().upper()
        language = str(payload.get("language", "en")).strip().lower()

        if not report_id or not symbol:
            logger.error("Invalid message payload received: %s", payload)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        logger.info("Received task | report_id=%s symbol=%s language=%s", report_id, symbol, language)

        report_record = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
        if not report_record:
            logger.warning("Database record not found for report_id=%s. Acknowledging message.", report_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        report_record.status = "PROCESSING"
        db.commit()

        cache_key = _build_cache_key(symbol, language)
        cached_report = None
        try:
            cached_report = redis_client.get(cache_key)
        except Exception as cache_err:
            logger.warning("Redis read skipped (%s): %s", cache_key, cache_err)

        if cached_report:
            logger.info("Cache HIT for %s (%s). Serving stored report.", symbol, language)
            report_text = cached_report
        else:
            logger.info("Cache MISS for %s (%s). Running AI research crew...", symbol, language)
            report_text = _run_analysis_heartbeat_safe(ch.connection, symbol, language)
            try:
                redis_client.setex(cache_key, CACHE_TTL_SECONDS, report_text)
                logger.info("Cached report in Redis under key=%s (TTL=%ss)", cache_key, CACHE_TTL_SECONDS)
            except Exception as cache_err:
                logger.warning("Redis write skipped (%s): %s", cache_key, cache_err)

        report_record.report = report_text
        report_record.status = "COMPLETED"
        db.commit()
        logger.info("Completed task | report_id=%s symbol=%s", report_id, symbol)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as exc:
        logger.exception("Failed processing analysis job: %s", exc)
        if report_record is not None:
            try:
                db.rollback()
                report_record.status = "FAILED"
                report_record.report = f"Analysis failed due to an internal error: {str(exc)}"
                db.commit()
            except Exception as db_err:
                logger.error("Failed to update FAILED status in PostgreSQL: %s", db_err)

        ch.basic_ack(delivery_tag=method.delivery_tag)
    finally:
        db.close()


def start_worker() -> None:
    """Connects to PostgreSQL and RabbitMQ and starts consuming analysis tasks."""
    init_db()
    logger.info("PostgreSQL tables verified.")

    while True:
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=process_analysis_job)

            logger.info("AlphaPulse Worker is online and listening on queue '%s'...", QUEUE_NAME)
            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Stopping AlphaPulse Worker...")
            break
        except pika.exceptions.AMQPConnectionError as conn_err:
            logger.warning("RabbitMQ unavailable (%s). Retrying in 5 seconds...", conn_err)
            time.sleep(5)


if __name__ == "__main__":
    start_worker()

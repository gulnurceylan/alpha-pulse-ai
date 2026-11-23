cat << 'EOF' > README.md
# AlphaPulse AI

**Asynchronous Multi-Agent Financial Research & Quantitative Analytics Platform**

AlphaPulse AI is an institutional-grade financial research system combining a deterministic Python quantitative engine (`pandas`, `numpy`) with an asynchronous multi-agent AI pipeline (`CrewAI`, `FastAPI`, `RabbitMQ`, `Redis`, `PostgreSQL`) and real-time Server-Sent Events (SSE) streaming.

## Key Engineering Highlights

- **Zero-Hallucination Quantitative Engine:** Deterministically computes RSI-14, MACD, Bollinger Bands, SMA 20/50/200, Annualized Volatility, Sharpe Ratio, Max Drawdown, valuation multiples, and 120-day OHLCV chart series before LLM synthesis.
- **Resilient Multi-Model Failover:** Stage-isolated CrewAI pipeline with bounded HTTP timeouts and automatic failover across prioritized Gemini models on upstream `429`/`503`/`504` errors.
- **Heartbeat-Safe Worker & Real-Time SSE:** RabbitMQ background consumer executes heavy AI workflows in a dedicated thread while maintaining AMQP heartbeats and streaming live stage events via Redis Pub/Sub + SSE.

## Development Roadmap

- [x] **Phase 1 — Core Async Backend:** FastAPI, PostgreSQL, RabbitMQ broker, Redis cache, and stage-isolated CrewAI agents.
- [x] **Phase 2 — Quantitative Engine & Live Streaming:** Deterministic technical/risk indicator engine (`app/quant_engine.py`) and Redis Pub/Sub + SSE progress streaming.
- [ ] **Phase 3 — Interactive Web Terminal (Next.js):** Dark-mode financial dashboard with TradingView candlestick charts, live agent timeline, and PDF export.
- [ ] **Phase 4 — Agentic RAG (`pgvector`):** Annual report & earnings PDF ingestion with citation-backed financial statement analysis.
- [ ] **Phase 5 — Head-to-Head Comparison & Watchlists:** Multi-ticker comparison mode, JWT auth, and automated alert triggers.

## Quick Start

1. `docker-compose up -d`
2. `source venv/bin/activate && python -m app.database`
3. `python -m app.worker` (Terminal 1)
4. `uvicorn app.main:app --reload` (Terminal 2)
5. Open `http://127.0.0.1:8000/docs`
EOF

git add .
git commit -m "feat: integrate deterministic quant engine, Redis SSE streaming, and v1 README"
git push origin main
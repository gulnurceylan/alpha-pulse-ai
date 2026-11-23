import json
import os
from collections.abc import Callable
from typing import Any

import yfinance as yf
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

from app.quant_engine import compute_quantitative_profile

load_dotenv()

# Priority failover chain: Always starts with the flagship model,
# switching immediately on upstream 503/429/504 overload errors.
MODEL_PRIORITY_CHAIN = [
    "gemini/gemini-4-hydrogen",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash",
]


def _extract_news_headlines(ticker: yf.Ticker, symbol: str, limit: int = 5) -> list[str]:
    """Extracts recent news headlines with publisher info using dual Yahoo endpoints."""
    headlines = []

    # Layer 1: Direct ticker news feed
    try:
        raw_news = ticker.news or []
        for item in raw_news:
            content = item.get("content") if isinstance(item.get("content"), dict) else item
            title = content.get("title")
            provider = content.get("provider", {})
            publisher = (
                provider.get("displayName")
                if isinstance(provider, dict)
                else item.get("publisher", "Market News")
            )
            if title:
                headlines.append(f"- [{publisher}] {title}")
            if len(headlines) >= limit:
                return headlines
    except Exception:
        pass

    # Layer 2: Fallback to Yahoo Search news index if ticker.news is empty
    try:
        search_results = yf.Search(symbol, news_count=limit).news or []
        for item in search_results:
            title = item.get("title")
            publisher = item.get("publisher", "Yahoo Finance")
            if title:
                headlines.append(f"- [{publisher}] {title}")
            if len(headlines) >= limit:
                break
    except Exception:
        pass

    return headlines


@tool("Market News and Catalysts Fetcher")
def fetch_market_news(symbol: str) -> str:
    """
    Fetches the latest market news headlines and catalysts for a given ticker symbol
    (e.g., AAPL, THYAO.IS, BTC-USD).
    """
    clean_symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        headlines = _extract_news_headlines(ticker, clean_symbol, limit=5)
        if not headlines:
            return f"No recent news headlines returned by provider for {clean_symbol}."
        return f"Latest News Headlines for {clean_symbol}:\n" + "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news headlines for {clean_symbol}: {str(e)}"


def _build_llm(model_name: str) -> LLM:
    """Initializes a Gemini LLM instance with bounded timeout for fast failover.

    client_params passes http_options directly to the underlying google-genai
    SDK Client:
      - timeout: 45 000 ms hard cap per request (unit is milliseconds in SDK)
      - retry_options.attempts=1: disables the SDK's built-in retry loop
        (which defaults to 5 attempts with exponential backoff, causing hangs
        of several minutes on 503/429). With attempts=1 the error surfaces
        immediately so _run_with_failover can switch to the next model at once.
    """
    return LLM(
        model=model_name,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        timeout=45,
        client_params={
            "http_options": {
                "timeout": 45_000,          # milliseconds; hard cap per HTTP call
                "retry_options": {"attempts": 1},  # 0 retries — fail fast
            }
        },
    )


def _run_with_failover(stage_name: str, task_runner) -> str:
    """
    Executes an isolated pipeline stage across MODEL_PRIORITY_CHAIN,
    failing over immediately on upstream 503/429/504 availability errors.
    """
    last_error = None
    for model_name in MODEL_PRIORITY_CHAIN:
        try:
            print(f"🤖 [{stage_name}] Running with model: {model_name}")
            return task_runner(_build_llm(model_name))
        except Exception as exc:
            last_error = exc
            error_msg = str(exc)
            if any(
                code in error_msg
                for code in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "504",
                    "DEADLINE_EXCEEDED",
                )
            ):
                print(
                    f"⚠️ [{stage_name}] {model_name} failed ({error_msg[:60].strip()}). Switching to next model..."
                )
                continue
            raise

    raise RuntimeError(f"[{stage_name}] All models in priority chain failed. Last error: {last_error}")


def _collect_market_briefing(symbol: str, quant_summary_json: str, llm: LLM) -> str:
    """Stage 1: Gathers live news headlines and combines them with deterministic quant metrics."""
    data_analyst = Agent(
        role="Senior Financial Data Analyst",
        goal=f"Retrieve recent news catalysts for {symbol} and structure them alongside verified quantitative metrics.",
        backstory=(
            "You are a meticulous Wall Street equity research analyst. "
            "You never guess or alter numerical outputs—you combine verified quantitative figures "
            "with real news catalysts into a clean briefing."
        ),
        tools=[fetch_market_news],
        llm=llm,
        max_iter=3,
        max_retry_limit=0,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Use the 'Market News and Catalysts Fetcher' tool once to retrieve the latest news headlines for '{symbol}'.\n"
            f"Combine those headlines with the deterministic quantitative profile below into a structured briefing:\n\n"
            f"{quant_summary_json}"
        ),
        expected_output=(
            "A complete structured briefing containing all verified fundamental multiples, technical indicators, "
            "risk metrics, and all retrieved news headlines."
        ),
        agent=data_analyst,
    )

    crew = Crew(
        agents=[data_analyst],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def _generate_strategy_report(symbol: str, language: str, briefing: str, llm: LLM) -> str:
    """Stage 2: Synthesizes the verified quantitative & news briefing into a 5-section institutional report."""
    target_language = "Turkish" if language.lower() == "tr" else "English"

    strategist = Agent(
        role="Chief Investment Strategist & Quantitative Risk Officer",
        goal=f"Deliver a comprehensive, deeply reasoned financial research report for {symbol} in {target_language}.",
        backstory=(
            "You are an institutional investment strategist who synthesizes deterministic technical indicators "
            "(RSI, MACD, Bollinger Bands, Moving Averages), risk-adjusted metrics (Sharpe Ratio, Max Drawdown), "
            "fundamental valuation multiples, and news sentiment into objective research reports."
        ),
        llm=llm,
        max_iter=2,
        max_retry_limit=0,
        verbose=True,
    )

    strategy_task = Task(
        description=(
            f"Using ONLY the verified quantitative and news briefing below for {symbol}, write a comprehensive, "
            f"institutional-grade financial analysis report in {target_language}.\n\n"
            f"--- VERIFIED QUANTITATIVE & NEWS BRIEFING ---\n{briefing}\n---------------------------------------------\n\n"
            "Structure the report clearly under these five sections:\n"
            "1. Executive Summary & Market Positioning\n"
            "2. Fundamental & Valuation Analysis (evaluate price level, 52-week positioning, P/E, P/B, EV/EBITDA, margins, and free cash flow)\n"
            "3. Technical & Momentum Analysis (interpret SMA 20/50/200 trend alignment, 14-day RSI regime, MACD signal, and Bollinger Bands)\n"
            "4. Quantitative Risk Profile (analyze annualized volatility, Sharpe ratio, and maximum drawdown)\n"
            "5. Market Sentiment, News Catalysts & Risk Factors (analyze news impact alongside balanced bull vs. bear drivers)\n\n"
            "Cite the exact numerical metrics from the briefing. Maintain an objective, analytical tone suitable for institutional research without issuing direct buy/sell advice."
        ),
        expected_output=f"A thorough, 5-section institutional financial & quantitative research report written in {target_language}.",
        agent=strategist,
    )

    crew = Crew(
        agents=[strategist],
        tasks=[strategy_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def run_financial_analysis(
    symbol: str,
    language: str = "en",
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """
    Orchestrates deterministic quantitative computation followed by the two-stage
    multi-agent pipeline with stage-isolated model failover and live progress hooks.
    """
    clean_symbol = symbol.strip().upper()

    # Step 0: Deterministic Quantitative Engine (Zero LLM math hallucination)
    quant_profile = compute_quantitative_profile(clean_symbol)
    if progress_callback:
        rsi_val = quant_profile["technical_indicators"]["rsi_14"]
        sharpe_val = quant_profile["risk_metrics"]["sharpe_ratio"]
        progress_callback(
            "QUANT_COMPUTED",
            f"Computed RSI ({rsi_val}), MACD, Sharpe Ratio ({sharpe_val}), and 120-day OHLCV chart series.",
        )

    # Keep LLM prompt focused by excluding the 120 raw chart candles
    quant_for_llm = {k: v for k, v in quant_profile.items() if k != "chart_series"}
    quant_summary_json = json.dumps(quant_for_llm, indent=2)

    # Step 1: News & Catalyst Research Agent
    if progress_callback:
        progress_callback(
            "STAGE_1_RESEARCH",
            "Senior Data Analyst Agent is collecting market news and aligning quantitative metrics...",
        )
    briefing = _run_with_failover(
        "Stage 1: Data Research",
        lambda llm: _collect_market_briefing(clean_symbol, quant_summary_json, llm),
    )

    # Step 2: Chief Investment Strategist Agent
    if progress_callback:
        progress_callback(
            "STAGE_2_STRATEGY",
            "Chief Investment Strategist Agent is writing the 5-section institutional report...",
        )
    report_text = _run_with_failover(
        "Stage 2: Strategy Report",
        lambda llm: _generate_strategy_report(clean_symbol, language, briefing, llm),
    )

    return {
        "report": report_text,
        "quant_data": quant_profile,
    }


if __name__ == "__main__":
    print("🚀 Running AlphaPulse AI Agents (Test: AAPL)...")
    result = run_financial_analysis(
        "AAPL",
        language="en",
        progress_callback=lambda step, msg: print(f"📡 [{step}] {msg}"),
    )
    print("\n" + "=" * 60)
    print("📊 GENERATED INSTITUTIONAL ANALYSIS REPORT:")
    print("=" * 60)
    print(result["report"])
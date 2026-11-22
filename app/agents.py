import os
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

load_dotenv()

# Priority failover chain: Always starts with the flagship model,
# switching immediately on upstream 503/429 overload errors.
MODEL_PRIORITY_CHAIN = [
    "gemini/gemini-4-hydrogen",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.6-flash",
    "gemini/gemini-3.5-flash",
]


def _format_large_number(value) -> str:
    """Formats large financial figures (e.g., Market Cap) into readable T/B/M units."""
    if not isinstance(value, (int, float)) or value <= 0:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.2f}"


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


@tool("Market Data and News Fetcher")
def fetch_market_data(symbol: str) -> str:
    """
    Fetches real-time price action, fundamental valuation ratios (P/E, Market Cap, Sector),
    and recent news headlines for a given ticker symbol (e.g., AAPL, THYAO.IS, BTC-USD).
    """
    clean_symbol = symbol.strip().upper()
    try:
        ticker = yf.Ticker(clean_symbol)
        fast = ticker.fast_info

        current_price = getattr(fast, "last_price", None)
        prev_close = getattr(fast, "previous_close", None)
        currency = getattr(fast, "currency", "USD")
        market_cap_raw = getattr(fast, "market_cap", None)
        high_52w = getattr(fast, "year_high", None)
        low_52w = getattr(fast, "year_low", None)

        daily_change = "N/A"
        if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
            pct = ((current_price - prev_close) / prev_close) * 100
            daily_change = f"{pct:+.2f}%"

        price_str = f"{current_price:.2f} {currency}" if isinstance(current_price, (int, float)) else "N/A"
        range_52w = (
            f"{low_52w:.2f} - {high_52w:.2f} {currency}"
            if isinstance(low_52w, (int, float)) and isinstance(high_52w, (int, float))
            else "N/A"
        )

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        company_name = info.get("longName") or info.get("shortName") or clean_symbol
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        profit_margin = info.get("profitMargins")

        trailing_pe_str = f"{trailing_pe:.2f}" if isinstance(trailing_pe, (int, float)) else "N/A"
        forward_pe_str = f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else "N/A"
        margin_str = f"{profit_margin * 100:.2f}%" if isinstance(profit_margin, (int, float)) else "N/A"

        headlines = _extract_news_headlines(ticker, clean_symbol, limit=5)
        news_section = "\n".join(headlines) if headlines else "No recent news headlines returned by provider."

        return (
            f"Asset: {company_name} ({clean_symbol})\n"
            f"Sector / Industry: {sector} | {industry}\n"
            f"Current Price: {price_str} (Daily Change: {daily_change})\n"
            f"Market Cap: {_format_large_number(market_cap_raw)}\n"
            f"Trailing P/E: {trailing_pe_str} | Forward P/E: {forward_pe_str}\n"
            f"Profit Margin: {margin_str}\n"
            f"52-Week Range: {range_52w}\n"
            f"Latest News Headlines:\n{news_section}"
        )
    except Exception as e:
        return f"Critical error fetching market data for {clean_symbol}: {str(e)}"


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

    Codes that trigger failover:
      - 503 UNAVAILABLE: model overloaded
      - 429 RESOURCE_EXHAUSTED: quota/rate limit hit
      - 504 DEADLINE_EXCEEDED: request timed out server-side (tenacity path in
        google-genai SDK surfaces this after the single attempt allowed by
        http_options.retry_options.attempts=1)
    """
    last_error = None
    for model_name in MODEL_PRIORITY_CHAIN:
        try:
            print(f"🤖 [{stage_name}] Running with model: {model_name}")
            return task_runner(_build_llm(model_name))
        except Exception as exc:
            last_error = exc
            error_msg = str(exc)
            if any(code in error_msg for code in (
                "503", "UNAVAILABLE",
                "429", "RESOURCE_EXHAUSTED",
                "504", "DEADLINE_EXCEEDED",
            )):
                print(f"⚠️ [{stage_name}] {model_name} failed ({error_msg[:60].strip()}). Switching to next model...")
                continue
            raise

    raise RuntimeError(f"[{stage_name}] All models in priority chain failed. Last error: {last_error}")


def _collect_market_briefing(symbol: str, llm: LLM) -> str:
    """Stage 1: Gathers live metrics, valuation ratios, and news headlines."""
    data_analyst = Agent(
        role="Senior Financial Data Analyst",
        goal=f"Collect and structure accurate market metrics, valuation ratios, and recent news for {symbol}.",
        backstory=(
            "You are a meticulous Wall Street equity research analyst. "
            "You extract exact market figures, valuation multiples, and news catalysts without speculation."
        ),
        tools=[fetch_market_data],
        llm=llm,
        max_iter=3,
        max_retry_limit=0,
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Use the 'Market Data and News Fetcher' tool once to retrieve the full financial profile, "
            f"valuation ratios, and latest news headlines for '{symbol}'. "
            "Present all retrieved metrics and headlines in a well-structured briefing."
        ),
        expected_output=(
            "A complete structured briefing containing company profile, price & daily change, "
            "market cap, Trailing/Forward P/E, profit margin, 52-week range, and all retrieved news headlines."
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
    """Stage 2: Synthesizes the verified market briefing into a 4-section institutional report."""
    target_language = "Turkish" if language.lower() == "tr" else "English"

    strategist = Agent(
        role="Chief Investment Strategist",
        goal=f"Deliver a comprehensive, deeply reasoned financial research report for {symbol} in {target_language}.",
        backstory=(
            "You are an institutional investment strategist who synthesizes fundamental valuation multiples, "
            "price momentum, and news sentiment into clear, objective research reports."
        ),
        llm=llm,
        max_iter=2,
        max_retry_limit=0,
        verbose=True,
    )

    strategy_task = Task(
        description=(
            f"Using the verified market briefing below for {symbol}, write a comprehensive, "
            f"institutional-grade financial analysis report in {target_language}.\n\n"
            f"--- VERIFIED MARKET BRIEFING ---\n{briefing}\n--------------------------------\n\n"
            "Structure the report clearly under these four sections:\n"
            "1. Overview & Valuation Analysis (evaluate price level, 52-week positioning, and P/E multiples)\n"
            "2. Market Sentiment & News Catalysts (analyze the impact of the retrieved news headlines)\n"
            "3. Key Opportunities & Risk Factors (balanced bull vs. bear drivers)\n"
            "4. Executive Summary\n"
            "Maintain an objective, analytical tone suitable for institutional research without issuing direct buy/sell advice."
        ),
        expected_output=f"A thorough, 4-section institutional financial analysis report written in {target_language}.",
        agent=strategist,
    )

    crew = Crew(
        agents=[strategist],
        tasks=[strategy_task],
        process=Process.sequential,
        verbose=True,
    )
    return str(crew.kickoff())


def run_financial_analysis(symbol: str, language: str = "en") -> str:
    """
    Orchestrates the two-stage financial analysis pipeline with stage-isolated
    multi-model failover for high availability.
    """
    briefing = _run_with_failover(
        "Stage 1: Data Research",
        lambda llm: _collect_market_briefing(symbol, llm),
    )
    return _run_with_failover(
        "Stage 2: Strategy Report",
        lambda llm: _generate_strategy_report(symbol, language, briefing, llm),
    )


if __name__ == "__main__":
    print("🚀 Running AlphaPulse AI Agents (Test: AAPL)...")
    report = run_financial_analysis("AAPL", language="en")
    print("\n" + "=" * 60)
    print("📊 GENERATED INSTITUTIONAL ANALYSIS REPORT:")
    print("=" * 60)
    print(report)
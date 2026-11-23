import math
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def _safe_float(value: Any, decimals: int = 2) -> float | None:
    """Converts numeric values safely to rounded floats, returning None for NaN/Inf."""
    if value is None:
        return None
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return round(num, decimals)
    except (TypeError, ValueError):
        return None


def compute_quantitative_profile(symbol: str, period: str = "1y") -> dict[str, Any]:
    """
    Fetches historical OHLCV data and company fundamentals via yfinance,
    then computes deterministic technical indicators, risk metrics, and chart series.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval="1d", auto_adjust=True)

    if hist.empty or len(hist) < 30:
        raise ValueError(f"Insufficient historical market data found for '{symbol}'.")

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    # 1. Moving Averages (Trend)
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None

    # 2. RSI (14-day Wilder's Smoothing)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_14 = rsi_series.iloc[-1]

    # 3. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # 4. Bollinger Bands (20-day, 2 std)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]

    # 5. Risk & Return Metrics (Annualized assuming 252 trading days)
    daily_returns = close.pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
    total_return_pct = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

    # Annualized Sharpe Ratio (assuming 4.25% risk-free rate)
    risk_free_rate = 0.0425
    annualized_return = daily_returns.mean() * 252
    annualized_std = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_std
        if annualized_std and annualized_std > 0
        else None
    )

    # Maximum Drawdown (%)
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100

    # 6. Fundamental Multiples (gracefully handles ETFs/Crypto where fields may be None)
    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    fundamentals = {
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency", "USD"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "profit_margin_pct": _safe_float((info.get("profitMargins") or 0) * 100)
        if info.get("profitMargins") is not None
        else None,
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "free_cash_flow": info.get("freeCashflow"),
        "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
    }

    # 7. Last 120 trading days OHLCV series ready for Frontend Candlestick Charts
    recent_hist = hist.tail(120)
    chart_series = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": _safe_float(row["Open"]),
            "high": _safe_float(row["High"]),
            "low": _safe_float(row["Low"]),
            "close": _safe_float(row["Close"]),
            "volume": int(row["Volume"]) if not math.isnan(row["Volume"]) else 0,
        }
        for idx, row in recent_hist.iterrows()
    ]

    latest_price = _safe_float(close.iloc[-1])
    prev_close = _safe_float(close.iloc[-2]) if len(close) >= 2 else latest_price
    daily_change_pct = (
        _safe_float(((latest_price - prev_close) / prev_close) * 100)
        if latest_price and prev_close
        else None
    )

    # Human-readable technical signal summary
    rsi_val = _safe_float(rsi_14)
    if rsi_val is not None:
        if rsi_val >= 70:
            rsi_signal = "Overbought (>=70)"
        elif rsi_val <= 30:
            rsi_signal = "Oversold (<=30)"
        else:
            rsi_signal = "Neutral (30-70)"
    else:
        rsi_signal = "Unavailable"

    return {
        "symbol": symbol,
        "latest_price": latest_price,
        "daily_change_pct": daily_change_pct,
        "fundamentals": fundamentals,
        "technical_indicators": {
            "sma_20": _safe_float(sma_20),
            "sma_50": _safe_float(sma_50),
            "sma_200": _safe_float(sma_200),
            "rsi_14": rsi_val,
            "rsi_regime": rsi_signal,
            "macd": _safe_float(macd_line.iloc[-1], 3),
            "macd_signal": _safe_float(macd_signal.iloc[-1], 3),
            "macd_histogram": _safe_float(macd_hist.iloc[-1], 3),
            "bollinger_upper": _safe_float(bb_upper),
            "bollinger_lower": _safe_float(bb_lower),
        },
        "risk_metrics": {
            "period_return_pct": _safe_float(total_return_pct),
            "annualized_volatility_pct": _safe_float(annualized_volatility),
            "sharpe_ratio": _safe_float(sharpe_ratio),
            "max_drawdown_pct": _safe_float(max_drawdown_pct),
        },
        "chart_series": chart_series,
    }


if __name__ == "__main__":
    import json

    result = compute_quantitative_profile("AAPL")
    preview = {k: v for k, v in result.items() if k != "chart_series"}
    preview["chart_points_count"] = len(result["chart_series"])
    print(json.dumps(preview, indent=2))
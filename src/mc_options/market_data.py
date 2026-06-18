"""Market option-chain helpers for listed European-style option analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from mc_options.black_scholes import black_scholes_price
from mc_options.implied_volatility import implied_volatility_from_price
from mc_options.utils import validate_option_type, validate_positive


def years_to_expiration(expiration: str, valuation_datetime: datetime | None = None) -> float:
    """Convert an option expiration date string into year fraction."""
    valuation_datetime = valuation_datetime or datetime.now(UTC)
    expiry_datetime = datetime.fromisoformat(expiration).replace(tzinfo=UTC)
    expiry_datetime = expiry_datetime.replace(hour=21, minute=0, second=0, microsecond=0)
    seconds = (expiry_datetime - valuation_datetime).total_seconds()
    return max(seconds / (365.0 * 24 * 60 * 60), 1 / 365)


def option_mid_price(row: pd.Series) -> float:
    """Return bid/ask midpoint when available, otherwise last traded price."""
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last_price = float(row.get("lastPrice", np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        return float((bid + ask) / 2)
    if np.isfinite(last_price) and last_price > 0:
        return float(last_price)
    return np.nan


def enrich_option_chain(
    chain: pd.DataFrame,
    S0: float,
    r: float,
    T: float,
    option_type: str,
    flat_sigma: float,
) -> pd.DataFrame:
    """Add mid prices, solved implied vols, and flat-vol model prices to a chain."""
    option_type = validate_option_type(option_type)
    validate_positive("S0", S0)
    validate_positive("T", T)
    validate_positive("flat_sigma", flat_sigma)

    rows: list[dict[str, Any]] = []
    for _, row in chain.iterrows():
        strike = float(row["strike"])
        mid = option_mid_price(row)
        solved_iv = np.nan
        model_price = black_scholes_price(S0, strike, r, flat_sigma, T, option_type)

        if np.isfinite(mid) and mid > 0:
            try:
                solved_iv = implied_volatility_from_price(mid, S0, strike, r, T, option_type)
            except ValueError:
                solved_iv = np.nan

        rows.append(
            {
                "contract_symbol": row.get("contractSymbol", ""),
                "strike": strike,
                "bid": float(row.get("bid", np.nan)),
                "ask": float(row.get("ask", np.nan)),
                "last_price": float(row.get("lastPrice", np.nan)),
                "mid_price": mid,
                "market_iv": float(row.get("impliedVolatility", np.nan)),
                "solved_iv": solved_iv,
                "flat_vol_model_price": model_price,
                "model_minus_mid": model_price - mid if np.isfinite(mid) else np.nan,
                "moneyness": strike / S0,
                "volume": row.get("volume", np.nan),
                "open_interest": row.get("openInterest", np.nan),
            }
        )

    enriched = pd.DataFrame(rows)
    return enriched.sort_values("strike").reset_index(drop=True)


def realized_volatility(
    prices: pd.Series,
    window: int,
    annualization: int = 252,
) -> float:
    """Compute annualized historical volatility from close prices."""
    validate_positive("window", window)
    clean_prices = prices.dropna().astype(float)
    if len(clean_prices) <= window:
        raise ValueError("not enough prices for requested volatility window")
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()
    window_returns = log_returns.tail(window)
    if len(window_returns) < window:
        raise ValueError("not enough returns for requested volatility window")
    return float(window_returns.std(ddof=1) * np.sqrt(annualization))


def realized_volatility_summary(
    prices: pd.Series,
    windows: tuple[int, ...] = (20, 60, 252),
    annualization: int = 252,
) -> pd.DataFrame:
    """Return annualized historical volatility for several return windows."""
    rows = []
    for window in windows:
        try:
            hv = realized_volatility(prices, window, annualization)
        except ValueError:
            hv = np.nan
        rows.append(
            {
                "window": f"{window}d",
                "trading_days": window,
                "historical_volatility": hv,
            }
        )
    return pd.DataFrame(rows)


def fetch_yfinance_option_snapshot(ticker: str) -> tuple[float, list[str]]:
    """Fetch spot price and available expirations from yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to use market option-chain mode") from exc

    symbol = yf.Ticker(ticker.upper())
    history = symbol.history(period="5d")
    if history.empty:
        raise ValueError(f"No recent price history returned for ticker {ticker!r}")
    spot = float(history["Close"].dropna().iloc[-1])
    expirations = list(symbol.options)
    if not expirations:
        raise ValueError(f"No listed option expirations returned for ticker {ticker!r}")
    return spot, expirations


def fetch_yfinance_option_chain(ticker: str, expiration: str, option_type: str) -> pd.DataFrame:
    """Fetch calls or puts for a ticker/expiration from yfinance."""
    option_type = validate_option_type(option_type)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to use market option-chain mode") from exc

    chain = yf.Ticker(ticker.upper()).option_chain(expiration)
    return chain.calls if option_type == "call" else chain.puts


def fetch_yfinance_close_prices(ticker: str, period: str = "18mo") -> pd.Series:
    """Fetch historical close prices for realized volatility analysis."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to use market option-chain mode") from exc

    history = yf.Ticker(ticker.upper()).history(period=period)
    if history.empty or "Close" not in history:
        raise ValueError(f"No historical close prices returned for ticker {ticker!r}")
    return history["Close"].dropna()

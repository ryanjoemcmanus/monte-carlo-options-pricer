"""Tests for option-chain enrichment helpers."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from mc_options.black_scholes import black_scholes_price
from mc_options.market_data import (
    enrich_option_chain,
    realized_volatility,
    realized_volatility_summary,
    years_to_expiration,
)


def test_years_to_expiration_returns_positive_fraction() -> None:
    valuation = datetime(2026, 1, 1, tzinfo=UTC)
    assert years_to_expiration("2026-07-01", valuation) > 0.49


def test_enrich_option_chain_adds_model_and_iv_columns() -> None:
    market_price = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "call")
    chain = pd.DataFrame(
        {
            "contractSymbol": ["TEST"],
            "strike": [100.0],
            "bid": [market_price - 0.05],
            "ask": [market_price + 0.05],
            "lastPrice": [market_price],
            "impliedVolatility": [0.20],
            "volume": [100],
            "openInterest": [500],
        }
    )
    enriched = enrich_option_chain(chain, 100, 0.05, 1.0, "call", flat_sigma=0.20)
    assert {"mid_price", "solved_iv", "flat_vol_model_price", "model_minus_mid"}.issubset(
        enriched.columns
    )
    assert float(enriched.loc[0, "solved_iv"]) == pytest.approx(0.20, abs=1e-9)


def test_realized_volatility_matches_manual_calculation() -> None:
    prices = pd.Series([100, 101, 99, 102, 104, 103, 105, 106, 104, 107, 108], dtype=float)
    hv = realized_volatility(prices, window=5, annualization=252)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    expected = float(log_returns.tail(5).std(ddof=1) * (252 ** 0.5))
    assert hv == pytest.approx(expected)


def test_realized_volatility_summary_returns_windows() -> None:
    prices = pd.Series(range(100, 400), dtype=float)
    summary = realized_volatility_summary(prices, windows=(20, 60))
    assert list(summary["window"]) == ["20d", "60d"]
    assert summary["historical_volatility"].notna().all()

"""Tests for implied volatility inversion."""

import pytest

from mc_options.black_scholes import black_scholes_price
from mc_options.implied_volatility import implied_volatility_from_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_recovers_known_sigma(option_type: str) -> None:
    price = black_scholes_price(100, 105, 0.04, 0.25, 0.75, option_type)
    implied_vol = implied_volatility_from_price(price, 100, 105, 0.04, 0.75, option_type)
    assert implied_vol == pytest.approx(0.25, abs=1e-7)


def test_implied_volatility_rejects_invalid_market_price() -> None:
    with pytest.raises(ValueError):
        implied_volatility_from_price(200, 100, 100, 0.05, 1.0, "call")

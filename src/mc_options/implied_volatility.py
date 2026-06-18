"""Implied volatility inversion for Black-Scholes option prices."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from mc_options.black_scholes import black_scholes_price
from mc_options.utils import validate_option_inputs, validate_option_type, validate_positive


def implied_volatility_from_price(
    market_price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    min_vol: float = 1e-6,
    max_vol: float = 5.0,
) -> float:
    """Solve for Black-Scholes implied volatility from an observed option price."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, min_vol, T)
    validate_positive("market_price", market_price)
    validate_positive("max_vol", max_vol)
    if max_vol <= min_vol:
        raise ValueError("max_vol must be greater than min_vol")

    intrinsic = max(S0 - K * math.exp(-r * T), 0.0)
    if option_type == "put":
        intrinsic = max(K * math.exp(-r * T) - S0, 0.0)
    upper_bound = S0 if option_type == "call" else K * math.exp(-r * T)

    if market_price < intrinsic - 1e-10:
        raise ValueError("market_price is below the no-arbitrage lower bound")
    if market_price > upper_bound + 1e-10:
        raise ValueError("market_price is above the no-arbitrage upper bound")

    def objective(volatility: float) -> float:
        return black_scholes_price(S0, K, r, volatility, T, option_type) - market_price

    low_value = objective(min_vol)
    high_value = objective(max_vol)
    if abs(low_value) < 1e-10:
        return float(min_vol)
    if abs(high_value) < 1e-10:
        return float(max_vol)
    if low_value * high_value > 0:
        raise ValueError("could not bracket implied volatility")

    return float(brentq(objective, min_vol, max_vol, xtol=1e-10, rtol=1e-10, maxiter=100))

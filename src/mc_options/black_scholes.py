"""Closed-form Black-Scholes prices and Greeks."""

from __future__ import annotations

import math

from scipy.stats import norm

from mc_options.utils import validate_option_inputs, validate_option_type


def _intrinsic_value(S0: float, K: float, option_type: str) -> float:
    if option_type == "call":
        return max(S0 - K, 0.0)
    return max(K - S0, 0.0)


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    """Price a European call or put under Black-Scholes.

    Degenerate cases with zero time or zero volatility are priced by discounting
    the deterministic terminal payoff under the risk-neutral drift.
    """
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)

    if T == 0:
        return float(_intrinsic_value(S0, K, option_type))

    if sigma == 0:
        deterministic_ST = S0 * math.exp(r * T)
        payoff = _intrinsic_value(deterministic_ST, K, option_type)
        return float(math.exp(-r * T) * payoff)

    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    discount = math.exp(-r * T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * discount * norm.cdf(d2)
    else:
        price = K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return float(price)


def black_scholes_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Return Black-Scholes Delta, Gamma, Vega, Theta, and Rho.

    Vega is per 1.00 change in volatility. Theta is annualized.
    """
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    if T == 0 or sigma == 0:
        raise ValueError("Black-Scholes Greeks require positive T and sigma")

    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    sqrt_T = math.sqrt(T)
    discount = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S0 * sigma * sqrt_T)
    vega = S0 * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -S0 * pdf_d1 * sigma / (2 * sqrt_T) - r * K * discount * norm.cdf(d2)
        rho = K * T * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -S0 * pdf_d1 * sigma / (2 * sqrt_T) + r * K * discount * norm.cdf(-d2)
        rho = -K * T * discount * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }

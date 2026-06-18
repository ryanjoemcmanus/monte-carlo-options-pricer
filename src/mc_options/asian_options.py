"""Asian option pricing under risk-neutral geometric Brownian motion."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from mc_options.monte_carlo import _summary_from_discounted_payoffs
from mc_options.utils import (
    validate_n_paths,
    validate_option_inputs,
    validate_option_type,
    validate_positive,
)


def validate_n_steps(n_steps: int) -> None:
    """Validate the number of monitoring dates in an Asian option."""
    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate GBM price paths at equally spaced monitoring dates.

    The returned array has shape ``(n_paths, n_steps)`` and excludes the initial
    price because fixed-strike Asian payoffs usually average monitored prices.
    """
    validate_option_inputs(S0, 1.0, r, sigma, T)
    validate_n_paths(n_paths)
    validate_n_steps(n_steps)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = math.log(S0) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def _gbm_paths_from_z(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_steps = z.shape[1]
    dt = T / n_steps
    increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = math.log(S0) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)


def _asian_payoff(
    average_prices: NDArray[np.float64], K: float, option_type: str
) -> NDArray[np.float64]:
    option_type = validate_option_type(option_type)
    if option_type == "call":
        return np.maximum(average_prices - K, 0.0)
    return np.maximum(K - average_prices, 0.0)


def _arithmetic_and_geometric_payoffs(
    paths: NDArray[np.float64], K: float, option_type: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    arithmetic_average = np.mean(paths, axis=1)
    geometric_average = np.exp(np.mean(np.log(paths), axis=1))
    return (
        _asian_payoff(arithmetic_average, K, option_type),
        _asian_payoff(geometric_average, K, option_type),
    )


def geometric_asian_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    option_type: str = "call",
) -> float:
    """Closed-form price for a discretely monitored geometric Asian option."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_steps(n_steps)

    if T == 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return float(intrinsic)

    if sigma == 0:
        monitoring_times = np.linspace(T / n_steps, T, n_steps)
        path = S0 * np.exp(r * monitoring_times)
        geometric_average = float(np.exp(np.mean(np.log(path))))
        payoff = (
            max(geometric_average - K, 0.0)
            if option_type == "call"
            else max(K - geometric_average, 0.0)
        )
        return float(math.exp(-r * T) * payoff)

    mean_log_average = math.log(S0) + (r - 0.5 * sigma**2) * T * (n_steps + 1) / (2 * n_steps)
    variance_log_average = (
        sigma**2 * T * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2)
    )
    volatility_log_average = math.sqrt(variance_log_average)
    d1 = (mean_log_average - math.log(K) + variance_log_average) / volatility_log_average
    d2 = d1 - volatility_log_average
    discount = math.exp(-r * T)
    forward_geometric_mean = math.exp(mean_log_average + 0.5 * variance_log_average)

    if option_type == "call":
        price = discount * (forward_geometric_mean * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount * (K * norm.cdf(-d2) - forward_geometric_mean * norm.cdf(-d1))
    return float(price)


def price_arithmetic_asian_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int | None = None,
) -> dict[str, Any]:
    """Price a fixed-strike arithmetic Asian option by plain Monte Carlo."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    arithmetic_payoffs, _ = _arithmetic_and_geometric_payoffs(paths, K, option_type)
    discounted_payoffs = math.exp(-r * T) * arithmetic_payoffs
    return _summary_from_discounted_payoffs(
        discounted_payoffs,
        option_type,
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        "asian_arithmetic_plain",
        {"n_steps": n_steps, "average_type": "arithmetic"},
    )


def price_arithmetic_asian_option_mc_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int | None = None,
) -> dict[str, Any]:
    """Price an arithmetic Asian option using geometric Asian payoff as control."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    arithmetic_payoffs, geometric_payoffs = _arithmetic_and_geometric_payoffs(paths, K, option_type)

    discount = math.exp(-r * T)
    discounted_arithmetic = discount * arithmetic_payoffs
    discounted_geometric = discount * geometric_payoffs
    geometric_price = geometric_asian_option_price(S0, K, r, sigma, T, n_steps, option_type)
    variance_control = float(np.var(discounted_geometric, ddof=1))
    beta = 0.0 if variance_control == 0 else float(
        np.cov(discounted_arithmetic, discounted_geometric, ddof=1)[0, 1] / variance_control
    )
    adjusted_discounted_payoffs = discounted_arithmetic - beta * (
        discounted_geometric - geometric_price
    )
    result = _summary_from_discounted_payoffs(
        adjusted_discounted_payoffs,
        option_type,
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        "asian_arithmetic_control_variate",
        {
            "n_steps": n_steps,
            "average_type": "arithmetic",
            "control": "geometric_asian",
            "geometric_asian_price": geometric_price,
            "beta": beta,
            "correlation": float(np.corrcoef(discounted_arithmetic, discounted_geometric)[0, 1]),
        },
    )
    return result


def _arithmetic_asian_price_from_z(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    z: NDArray[np.float64],
    option_type: str,
    use_control_variate: bool,
) -> float:
    paths = _gbm_paths_from_z(S0, r, sigma, T, z)
    arithmetic_payoffs, geometric_payoffs = _arithmetic_and_geometric_payoffs(paths, K, option_type)
    discount = math.exp(-r * T)
    discounted_arithmetic = discount * arithmetic_payoffs

    if not use_control_variate:
        return float(np.mean(discounted_arithmetic))

    discounted_geometric = discount * geometric_payoffs
    geometric_price = geometric_asian_option_price(S0, K, r, sigma, T, z.shape[1], option_type)
    variance_control = float(np.var(discounted_geometric, ddof=1))
    beta = 0.0 if variance_control == 0 else float(
        np.cov(discounted_arithmetic, discounted_geometric, ddof=1)[0, 1] / variance_control
    )
    adjusted_discounted = discounted_arithmetic - beta * (discounted_geometric - geometric_price)
    return float(np.mean(adjusted_discounted))


def estimate_asian_greeks_mc_finite_difference(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int | None = None,
    bump_pct: float = 0.01,
    use_control_variate: bool = True,
) -> dict[str, float | int | str | bool]:
    """Estimate arithmetic Asian option Greeks with finite differences.

    Common random numbers are used across bumped full-path simulations so the
    Greek estimates respond to parameter changes rather than fresh MC noise.
    """
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)
    validate_n_steps(n_steps)
    validate_positive("bump_pct", bump_pct)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    base_price = _arithmetic_asian_price_from_z(
        S0, K, r, sigma, T, z, option_type, use_control_variate
    )

    dS = S0 * bump_pct
    price_s_up = _arithmetic_asian_price_from_z(
        S0 + dS, K, r, sigma, T, z, option_type, use_control_variate
    )
    price_s_down = _arithmetic_asian_price_from_z(
        max(S0 - dS, 1e-12), K, r, sigma, T, z, option_type, use_control_variate
    )
    delta = (price_s_up - price_s_down) / (2 * dS)
    gamma = (price_s_up - 2 * base_price + price_s_down) / (dS**2)

    d_sigma = max(sigma * bump_pct, 1e-4)
    price_vol_up = _arithmetic_asian_price_from_z(
        S0, K, r, sigma + d_sigma, T, z, option_type, use_control_variate
    )
    price_vol_down = _arithmetic_asian_price_from_z(
        S0, K, r, max(sigma - d_sigma, 1e-12), T, z, option_type, use_control_variate
    )
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

    dr = max(abs(r) * bump_pct, 1e-4)
    price_r_up = _arithmetic_asian_price_from_z(
        S0, K, r + dr, sigma, T, z, option_type, use_control_variate
    )
    price_r_down = _arithmetic_asian_price_from_z(
        S0, K, r - dr, sigma, T, z, option_type, use_control_variate
    )
    rho = (price_r_up - price_r_down) / (2 * dr)

    dT = min(max(T * bump_pct, 1e-4), T * 0.5)
    price_t_up = _arithmetic_asian_price_from_z(
        S0, K, r, sigma, T + dT, z, option_type, use_control_variate
    )
    price_t_down = _arithmetic_asian_price_from_z(
        S0, K, r, sigma, T - dT, z, option_type, use_control_variate
    )
    theta = -(price_t_up - price_t_down) / (2 * dT)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "method": "asian_finite_difference_mc",
        "n_paths": n_paths,
        "n_steps": n_steps,
        "bump_pct": bump_pct,
        "use_control_variate": use_control_variate,
    }

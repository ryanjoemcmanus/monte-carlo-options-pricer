"""Monte Carlo pricing engines for European options."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, qmc

from mc_options.utils import (
    european_payoff,
    validate_n_paths,
    validate_option_inputs,
    validate_option_type,
    validate_positive,
)


def _terminal_prices_from_z(
    S0: float, r: float, sigma: float, T: float, z: NDArray[np.float64]
) -> NDArray[np.float64]:
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)


def simulate_terminal_prices(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate terminal stock prices under risk-neutral GBM."""
    validate_option_inputs(S0, 1.0, r, sigma, T)
    validate_n_paths(n_paths)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return _terminal_prices_from_z(S0, r, sigma, T, z)


def simulate_terminal_prices_sobol(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
    scramble: bool = True,
) -> NDArray[np.float64]:
    """Simulate terminal GBM prices from Sobol quasi-random normal draws."""
    validate_option_inputs(S0, 1.0, r, sigma, T)
    validate_n_paths(n_paths)

    sampler = qmc.Sobol(d=1, scramble=scramble, seed=seed)
    m = math.ceil(math.log2(n_paths))
    uniforms = sampler.random_base2(m=m)[:n_paths, 0]
    uniforms = np.clip(uniforms, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    z = norm.ppf(uniforms)
    return _terminal_prices_from_z(S0, r, sigma, T, z)


def _summary_from_discounted_payoffs(
    discounted_payoffs: NDArray[np.float64],
    option_type: str,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    method: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    price = float(np.mean(discounted_payoffs))
    payoff_std = float(np.std(discounted_payoffs, ddof=1))
    standard_error = payoff_std / math.sqrt(len(discounted_payoffs))
    result: dict[str, Any] = {
        "option_type": option_type,
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n_paths": n_paths,
        "mc_price": price,
        "standard_error": float(standard_error),
        "ci_lower": float(price - 1.96 * standard_error),
        "ci_upper": float(price + 1.96 * standard_error),
        "discounted_payoff_std": payoff_std,
        "method": method,
    }
    if extras:
        result.update(extras)
    return result


def price_european_option_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict[str, Any]:
    """Price a European option by plain Monte Carlo simulation."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)

    terminal_prices = simulate_terminal_prices(S0, r, sigma, T, n_paths, seed)
    discounted_payoffs = math.exp(-r * T) * european_payoff(terminal_prices, K, option_type)
    return _summary_from_discounted_payoffs(
        discounted_payoffs, option_type, S0, K, r, sigma, T, n_paths, "plain"
    )


def price_european_option_mc_antithetic(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict[str, Any]:
    """Price a European option using antithetic variates."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)
    if n_paths % 2 != 0:
        raise ValueError("Antithetic variates require an even n_paths value")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    terminal_up = _terminal_prices_from_z(S0, r, sigma, T, z)
    terminal_down = _terminal_prices_from_z(S0, r, sigma, T, -z)
    payoffs_up = european_payoff(terminal_up, K, option_type)
    payoffs_down = european_payoff(terminal_down, K, option_type)
    paired_discounted_payoffs = math.exp(-r * T) * 0.5 * (payoffs_up + payoffs_down)

    result = _summary_from_discounted_payoffs(
        paired_discounted_payoffs, option_type, S0, K, r, sigma, T, n_paths, "antithetic"
    )
    result["standard_error"] = float(result["discounted_payoff_std"] / math.sqrt(n_paths // 2))
    result["ci_lower"] = float(result["mc_price"] - 1.96 * result["standard_error"])
    result["ci_upper"] = float(result["mc_price"] + 1.96 * result["standard_error"])
    return result


def price_european_option_mc_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
) -> dict[str, Any]:
    """Price a European option using terminal stock price as a control variate."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)

    terminal_prices = simulate_terminal_prices(S0, r, sigma, T, n_paths, seed)
    payoffs = european_payoff(terminal_prices, K, option_type)
    expected_terminal_price = S0 * math.exp(r * T)
    variance_control = float(np.var(terminal_prices, ddof=1))
    beta = 0.0 if variance_control == 0 else float(
        np.cov(payoffs, terminal_prices, ddof=1)[0, 1] / variance_control
    )
    adjusted_payoffs = payoffs - beta * (terminal_prices - expected_terminal_price)
    discounted_adjusted_payoffs = math.exp(-r * T) * adjusted_payoffs
    return _summary_from_discounted_payoffs(
        discounted_adjusted_payoffs,
        option_type,
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        "control_variate",
        {"beta": beta},
    )


def price_european_option_mc_sobol(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
    scramble: bool = True,
) -> dict[str, Any]:
    """Price a European option with scrambled Sobol quasi-Monte Carlo."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)

    terminal_prices = simulate_terminal_prices_sobol(
        S0, r, sigma, T, n_paths, seed=seed, scramble=scramble
    )
    discounted_payoffs = math.exp(-r * T) * european_payoff(terminal_prices, K, option_type)
    generated_paths = 2 ** math.ceil(math.log2(n_paths))
    return _summary_from_discounted_payoffs(
        discounted_payoffs,
        option_type,
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        "sobol_quasi_mc",
        {
            "scramble": scramble,
            "sobol_dimension": 1,
            "sobol_generated_paths": generated_paths,
        },
    )


def _mc_price_from_z(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    z: NDArray[np.float64],
    option_type: str,
) -> float:
    terminal_prices = _terminal_prices_from_z(S0, r, sigma, T, z)
    payoffs = european_payoff(terminal_prices, K, option_type)
    return float(math.exp(-r * T) * np.mean(payoffs))


def estimate_greeks_mc_finite_difference(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
    bump_pct: float = 0.01,
) -> dict[str, float | int | str]:
    """Estimate Greeks with central finite differences and common random numbers."""
    option_type = validate_option_type(option_type)
    validate_option_inputs(S0, K, r, sigma, T)
    validate_n_paths(n_paths)
    validate_positive("bump_pct", bump_pct)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    base_price = _mc_price_from_z(S0, K, r, sigma, T, z, option_type)

    dS = S0 * bump_pct
    price_s_up = _mc_price_from_z(S0 + dS, K, r, sigma, T, z, option_type)
    price_s_down = _mc_price_from_z(max(S0 - dS, 1e-12), K, r, sigma, T, z, option_type)
    delta = (price_s_up - price_s_down) / (2 * dS)
    gamma = (price_s_up - 2 * base_price + price_s_down) / (dS**2)

    d_sigma = max(sigma * bump_pct, 1e-4)
    price_vol_up = _mc_price_from_z(S0, K, r, sigma + d_sigma, T, z, option_type)
    price_vol_down = _mc_price_from_z(S0, K, r, max(sigma - d_sigma, 1e-12), T, z, option_type)
    vega = (price_vol_up - price_vol_down) / (2 * d_sigma)

    dr = max(abs(r) * bump_pct, 1e-4)
    price_r_up = _mc_price_from_z(S0, K, r + dr, sigma, T, z, option_type)
    price_r_down = _mc_price_from_z(S0, K, r - dr, sigma, T, z, option_type)
    rho = (price_r_up - price_r_down) / (2 * dr)

    dT = min(max(T * bump_pct, 1e-4), T * 0.5)
    price_t_up = _mc_price_from_z(S0, K, r, sigma, T + dT, z, option_type)
    price_t_down = _mc_price_from_z(S0, K, r, sigma, T - dT, z, option_type)
    theta = -(price_t_up - price_t_down) / (2 * dT)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "method": "finite_difference_mc",
        "n_paths": n_paths,
        "bump_pct": bump_pct,
    }

"""Experiment helpers for pricing accuracy, convergence, and Greeks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mc_options.asian_options import (
    estimate_asian_greeks_mc_finite_difference,
    geometric_asian_option_price,
    price_arithmetic_asian_option_mc,
    price_arithmetic_asian_option_mc_control_variate,
)
from mc_options.black_scholes import black_scholes_greeks, black_scholes_price
from mc_options.monte_carlo import (
    estimate_greeks_mc_finite_difference,
    price_european_option_mc,
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
)
from mc_options.utils import validate_option_type


def _relative_error_pct(estimate: float, benchmark: float) -> float:
    if benchmark == 0:
        return np.nan
    return abs(estimate - benchmark) / abs(benchmark) * 100


def run_convergence_experiment(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    path_counts: list[int],
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Monte Carlo estimates to Black-Scholes as path counts increase."""
    option_type = validate_option_type(option_type)
    bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
    rows = []

    for n_paths in path_counts:
        trial_prices = [
            price_european_option_mc(S0, K, r, sigma, T, n_paths, option_type, seed + i)[
                "mc_price"
            ]
            for i in range(n_trials)
        ]
        mean_price = float(np.mean(trial_prices))
        rows.append(
            {
                "n_paths": n_paths,
                "mean_mc_price": mean_price,
                "std_mc_price": float(np.std(trial_prices, ddof=1)),
                "black_scholes_price": bs_price,
                "absolute_error": abs(mean_price - bs_price),
                "relative_error_pct": _relative_error_pct(mean_price, bs_price),
            }
        )

    return pd.DataFrame(rows)


def run_moneyness_experiment(
    S0_values: list[float],
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare MC and Black-Scholes prices across underlying spot values."""
    option_type = validate_option_type(option_type)
    rows = []

    for idx, S0 in enumerate(S0_values):
        mc_result = price_european_option_mc(S0, K, r, sigma, T, n_paths, option_type, seed + idx)
        bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
        rows.append(
            {
                "S0": S0,
                "K": K,
                "moneyness": S0 / K,
                "mc_price": mc_result["mc_price"],
                "standard_error": mc_result["standard_error"],
                "black_scholes_price": bs_price,
                "absolute_error": abs(mc_result["mc_price"] - bs_price),
                "relative_error_pct": _relative_error_pct(mc_result["mc_price"], bs_price),
            }
        )

    return pd.DataFrame(rows)


def run_variance_reduction_experiment(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int = 100_000,
    n_trials: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare plain MC, antithetic variates, and control variates."""
    option_type = validate_option_type(option_type)
    bs_price = black_scholes_price(S0, K, r, sigma, T, option_type)
    methods = {
        "plain": price_european_option_mc,
        "antithetic": price_european_option_mc_antithetic,
        "control_variate": price_european_option_mc_control_variate,
    }
    rows = []

    for method_name, method_func in methods.items():
        results = [
            method_func(S0, K, r, sigma, T, n_paths, option_type, seed + i)
            for i in range(n_trials)
        ]
        prices = np.array([result["mc_price"] for result in results])
        standard_errors = np.array([result["standard_error"] for result in results])
        mean_price = float(np.mean(prices))
        rows.append(
            {
                "method": method_name,
                "mean_price": mean_price,
                "std_price_across_trials": float(np.std(prices, ddof=1)),
                "mean_standard_error": float(np.mean(standard_errors)),
                "black_scholes_price": bs_price,
                "absolute_error": abs(mean_price - bs_price),
                "relative_error_pct": _relative_error_pct(mean_price, bs_price),
            }
        )

    return pd.DataFrame(rows)


def run_greeks_comparison(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int = 500_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare closed-form Greeks against MC finite-difference estimates."""
    option_type = validate_option_type(option_type)
    bs_greeks = black_scholes_greeks(S0, K, r, sigma, T, option_type)
    mc_greeks = estimate_greeks_mc_finite_difference(
        S0, K, r, sigma, T, n_paths, option_type, seed
    )
    rows = []

    for greek, bs_value in bs_greeks.items():
        mc_value = float(mc_greeks[greek])
        rows.append(
            {
                "greek": greek,
                "black_scholes_value": bs_value,
                "monte_carlo_estimate": mc_value,
                "absolute_error": abs(mc_value - bs_value),
                "relative_error_pct": _relative_error_pct(mc_value, bs_value),
            }
        )

    return pd.DataFrame(rows)


def run_asian_option_experiment(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int = 100_000,
    n_steps: int = 252,
    n_trials: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare arithmetic Asian MC methods and geometric Asian benchmark."""
    option_type = validate_option_type(option_type)
    geometric_price = geometric_asian_option_price(S0, K, r, sigma, T, n_steps, option_type)
    methods = {
        "arithmetic_plain": price_arithmetic_asian_option_mc,
        "arithmetic_geometric_control": price_arithmetic_asian_option_mc_control_variate,
    }
    rows = []

    for method_name, method_func in methods.items():
        results = [
            method_func(S0, K, r, sigma, T, n_paths, n_steps, option_type, seed + i)
            for i in range(n_trials)
        ]
        prices = np.array([result["mc_price"] for result in results])
        standard_errors = np.array([result["standard_error"] for result in results])
        row = {
            "method": method_name,
            "mean_price": float(np.mean(prices)),
            "std_price_across_trials": float(np.std(prices, ddof=1)),
            "mean_standard_error": float(np.mean(standard_errors)),
            "geometric_asian_price": geometric_price,
            "n_paths": n_paths,
            "n_steps": n_steps,
        }
        if method_name == "arithmetic_geometric_control":
            row["mean_beta"] = float(np.mean([result["beta"] for result in results]))
            row["mean_correlation"] = float(np.mean([result["correlation"] for result in results]))
        else:
            row["mean_beta"] = np.nan
            row["mean_correlation"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def run_asian_greeks_comparison(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare plain and control-variate finite-difference Asian Greeks."""
    option_type = validate_option_type(option_type)
    plain = estimate_asian_greeks_mc_finite_difference(
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        n_steps,
        option_type,
        seed,
        use_control_variate=False,
    )
    control = estimate_asian_greeks_mc_finite_difference(
        S0,
        K,
        r,
        sigma,
        T,
        n_paths,
        n_steps,
        option_type,
        seed,
        use_control_variate=True,
    )
    rows = []
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        plain_value = float(plain[greek])
        control_value = float(control[greek])
        rows.append(
            {
                "greek": greek,
                "plain_mc_estimate": plain_value,
                "control_variate_estimate": control_value,
                "absolute_difference": abs(control_value - plain_value),
                "n_paths": n_paths,
                "n_steps": n_steps,
            }
        )
    return pd.DataFrame(rows)

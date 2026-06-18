"""Tests for Monte Carlo pricing engines."""

import numpy as np
import pytest

from mc_options.black_scholes import black_scholes_greeks, black_scholes_price
from mc_options.monte_carlo import (
    estimate_greeks_mc_finite_difference,
    price_european_option_mc,
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
    price_european_option_mc_sobol,
    simulate_terminal_prices,
    simulate_terminal_prices_sobol,
)


def test_terminal_price_shape_and_seed_reproducibility() -> None:
    first = simulate_terminal_prices(100, 0.05, 0.20, 1.0, 10_000, seed=7)
    second = simulate_terminal_prices(100, 0.05, 0.20, 1.0, 10_000, seed=7)
    assert len(first) == 10_000
    np.testing.assert_allclose(first, second)


def test_sobol_terminal_prices_are_reproducible() -> None:
    first = simulate_terminal_prices_sobol(100, 0.05, 0.20, 1.0, 10_000, seed=7)
    second = simulate_terminal_prices_sobol(100, 0.05, 0.20, 1.0, 10_000, seed=7)
    assert len(first) == 10_000
    np.testing.assert_allclose(first, second)


def test_mc_price_close_to_black_scholes_and_ci_contains_benchmark() -> None:
    result = price_european_option_mc(100, 100, 0.05, 0.20, 1.0, 500_000, "call", seed=42)
    bs_price = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "call")
    assert result["mc_price"] == pytest.approx(bs_price, abs=0.10)
    assert result["ci_lower"] <= bs_price <= result["ci_upper"]


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_antithetic_returns_valid_price(option_type: str) -> None:
    result = price_european_option_mc_antithetic(
        100, 100, 0.05, 0.20, 1.0, 100_000, option_type, seed=42
    )
    assert result["mc_price"] > 0
    assert result["standard_error"] >= 0
    assert result["method"] == "antithetic"


def test_antithetic_requires_even_paths() -> None:
    with pytest.raises(ValueError):
        price_european_option_mc_antithetic(100, 100, 0.05, 0.20, 1.0, 99_999, "call", 42)


def test_control_variate_returns_beta_and_close_price() -> None:
    result = price_european_option_mc_control_variate(
        100, 100, 0.05, 0.20, 1.0, 200_000, "call", seed=42
    )
    bs_price = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "call")
    assert "beta" in result
    assert result["standard_error"] >= 0
    assert result["mc_price"] == pytest.approx(bs_price, abs=0.08)


def test_sobol_quasi_mc_returns_close_price() -> None:
    result = price_european_option_mc_sobol(
        100, 100, 0.05, 0.20, 1.0, 4_096, "call", seed=42
    )
    bs_price = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "call")
    assert result["method"] == "sobol_quasi_mc"
    assert result["sobol_dimension"] == 1
    assert result["sobol_generated_paths"] == 4_096
    assert result["mc_price"] == pytest.approx(bs_price, abs=0.02)


def test_finite_difference_greeks_keys_and_delta_accuracy() -> None:
    mc_greeks = estimate_greeks_mc_finite_difference(
        100, 100, 0.05, 0.20, 1.0, 300_000, "call", seed=42
    )
    bs_greeks = black_scholes_greeks(100, 100, 0.05, 0.20, 1.0, "call")
    assert {"delta", "gamma", "vega", "theta", "rho"}.issubset(mc_greeks)
    assert mc_greeks["delta"] == pytest.approx(bs_greeks["delta"], abs=0.03)

"""Tests for Asian option pricing."""

import numpy as np
import pytest

from mc_options.asian_options import (
    estimate_asian_greeks_mc_finite_difference,
    geometric_asian_option_price,
    price_arithmetic_asian_option_mc,
    price_arithmetic_asian_option_mc_control_variate,
    simulate_gbm_paths,
)


def test_simulate_gbm_paths_shape_and_seed_reproducibility() -> None:
    first = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 1_000, 12, seed=42)
    second = simulate_gbm_paths(100, 0.05, 0.20, 1.0, 1_000, 12, seed=42)
    assert first.shape == (1_000, 12)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_geometric_asian_price_positive(option_type: str) -> None:
    price = geometric_asian_option_price(100, 100, 0.05, 0.20, 1.0, 252, option_type)
    assert price > 0


def test_arithmetic_asian_mc_returns_valid_call_price() -> None:
    result = price_arithmetic_asian_option_mc(
        100, 100, 0.05, 0.20, 1.0, 100_000, 252, "call", seed=42
    )
    assert result["mc_price"] > 0
    assert result["standard_error"] > 0
    assert result["average_type"] == "arithmetic"
    assert result["n_steps"] == 252


def test_geometric_control_variate_reduces_standard_error() -> None:
    plain = price_arithmetic_asian_option_mc(
        100, 100, 0.05, 0.20, 1.0, 100_000, 252, "call", seed=42
    )
    control = price_arithmetic_asian_option_mc_control_variate(
        100, 100, 0.05, 0.20, 1.0, 100_000, 252, "call", seed=42
    )
    assert control["standard_error"] < plain["standard_error"]
    assert control["correlation"] > 0.99
    assert "beta" in control


def test_invalid_n_steps_raises() -> None:
    with pytest.raises(ValueError):
        price_arithmetic_asian_option_mc(100, 100, 0.05, 0.20, 1.0, 10_000, 0, "call", 42)


def test_asian_finite_difference_greeks_have_expected_keys_and_ranges() -> None:
    greeks = estimate_asian_greeks_mc_finite_difference(
        100, 100, 0.05, 0.20, 1.0, 50_000, 252, "call", seed=42
    )
    assert {"delta", "gamma", "vega", "theta", "rho"}.issubset(greeks)
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    assert greeks["rho"] > 0
    assert greeks["use_control_variate"] is True


def test_asian_put_delta_range() -> None:
    greeks = estimate_asian_greeks_mc_finite_difference(
        100, 100, 0.05, 0.20, 1.0, 50_000, 252, "put", seed=42
    )
    assert -1 < greeks["delta"] < 0

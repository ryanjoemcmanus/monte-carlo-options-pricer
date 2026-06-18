"""Tests for Black-Scholes prices and Greeks."""

import pytest

from mc_options.black_scholes import black_scholes_greeks, black_scholes_price


def test_black_scholes_known_atm_values() -> None:
    call = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "call")
    put = black_scholes_price(100, 100, 0.05, 0.20, 1.0, "put")
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 0.05, 0.20, 1.0, "straddle")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S0": -100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0},
        {"S0": 100, "K": -100, "r": 0.05, "sigma": 0.20, "T": 1.0},
        {"S0": 100, "K": 100, "r": 0.05, "sigma": -0.20, "T": 1.0},
        {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": -1.0},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        black_scholes_price(**kwargs)


def test_black_scholes_greeks_have_expected_shape() -> None:
    greeks = black_scholes_greeks(100, 100, 0.05, 0.20, 1.0, "call")
    assert set(greeks) == {"delta", "gamma", "vega", "theta", "rho"}
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


def test_put_delta_range() -> None:
    greeks = black_scholes_greeks(100, 100, 0.05, 0.20, 1.0, "put")
    assert -1 < greeks["delta"] < 0

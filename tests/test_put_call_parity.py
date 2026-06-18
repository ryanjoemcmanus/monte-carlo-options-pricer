"""Put-call parity tests."""

import math

import pytest

from mc_options.black_scholes import black_scholes_price


def test_put_call_parity() -> None:
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.20
    T = 1.0
    call = black_scholes_price(S0, K, r, sigma, T, "call")
    put = black_scholes_price(S0, K, r, sigma, T, "put")
    parity_rhs = S0 - K * math.exp(-r * T)
    assert call - put == pytest.approx(parity_rhs, abs=1e-10)

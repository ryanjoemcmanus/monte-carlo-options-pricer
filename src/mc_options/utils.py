"""Shared validation and payoff helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


VALID_OPTION_TYPES = {"call", "put"}


def validate_option_type(option_type: str) -> str:
    """Return a normalized option type or raise ValueError."""
    normalized = option_type.lower()
    if normalized not in VALID_OPTION_TYPES:
        raise ValueError("option_type must be either 'call' or 'put'")
    return normalized


def validate_positive(name: str, value: float) -> None:
    """Validate that a scalar input is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_nonnegative(name: str, value: float) -> None:
    """Validate that a scalar input is nonnegative."""
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_option_inputs(S0: float, K: float, r: float, sigma: float, T: float) -> None:
    """Validate common option pricing inputs."""
    validate_positive("S0", S0)
    validate_positive("K", K)
    validate_nonnegative("sigma", sigma)
    validate_nonnegative("T", T)


def validate_n_paths(n_paths: int) -> None:
    """Validate a Monte Carlo path count."""
    if not isinstance(n_paths, int):
        raise TypeError("n_paths must be an integer")
    if n_paths <= 1:
        raise ValueError("n_paths must be greater than 1")


def european_payoff(
    terminal_prices: NDArray[np.float64], K: float, option_type: str
) -> NDArray[np.float64]:
    """Compute European call or put payoffs from terminal prices."""
    option_type = validate_option_type(option_type)
    if option_type == "call":
        return np.maximum(terminal_prices - K, 0.0)
    return np.maximum(K - terminal_prices, 0.0)

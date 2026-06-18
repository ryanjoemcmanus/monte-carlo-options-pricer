"""Matplotlib plotting helpers for report outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_if_requested(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")


def plot_convergence(results_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot Monte Carlo mean price against the Black-Scholes benchmark."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["n_paths"], results_df["mean_mc_price"], marker="o", label="Monte Carlo")
    ax.axhline(
        results_df["black_scholes_price"].iloc[0],
        color="black",
        linestyle="--",
        label="Black-Scholes",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Option price")
    ax.set_title("Monte Carlo Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_if_requested(fig, output_path)
    return fig


def plot_error_vs_paths(results_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot absolute pricing error against path count."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["n_paths"], results_df["absolute_error"], marker="o", color="tab:red")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Absolute error")
    ax.set_title("Pricing Error vs. Monte Carlo Paths")
    ax.grid(True, alpha=0.3)
    _save_if_requested(fig, output_path)
    return fig


def plot_moneyness_comparison(
    results_df: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot MC and Black-Scholes prices across moneyness."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["moneyness"], results_df["mc_price"], marker="o", label="Monte Carlo")
    ax.plot(
        results_df["moneyness"],
        results_df["black_scholes_price"],
        marker="s",
        linestyle="--",
        label="Black-Scholes",
    )
    ax.set_xlabel("Moneyness (S0 / K)")
    ax.set_ylabel("Option price")
    ax.set_title("Price Comparison Across Moneyness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_if_requested(fig, output_path)
    return fig


def plot_variance_reduction_comparison(
    results_df: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Create a bar chart comparing standard errors by Monte Carlo method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df["method"], results_df["mean_standard_error"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean standard error")
    ax.set_title("Variance Reduction Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    _save_if_requested(fig, output_path)
    return fig


def plot_greeks_comparison(
    results_df: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot Black-Scholes and MC finite-difference Greek estimates."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(results_df))
    width = 0.35
    ax.bar(
        [idx - width / 2 for idx in x],
        results_df["black_scholes_value"],
        width=width,
        label="Black-Scholes",
    )
    ax.bar(
        [idx + width / 2 for idx in x],
        results_df["monte_carlo_estimate"],
        width=width,
        label="Monte Carlo",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["greek"].str.title())
    ax.set_ylabel("Greek value")
    ax.set_title("Greeks: Black-Scholes vs. Monte Carlo")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_if_requested(fig, output_path)
    return fig


def plot_asian_option_methods(
    results_df: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot arithmetic Asian option standard errors by pricing method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        results_df["method"],
        results_df["mean_standard_error"],
        color=["#4C78A8", "#54A24B"],
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean standard error")
    ax.set_title("Arithmetic Asian Option Variance Reduction")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15)
    _save_if_requested(fig, output_path)
    return fig


def plot_asian_greeks_comparison(
    results_df: pd.DataFrame, output_path: str | Path | None = None
) -> plt.Figure:
    """Plot plain and control-variate Asian option Greek estimates."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(results_df))
    width = 0.35
    ax.bar(
        [idx - width / 2 for idx in x],
        results_df["plain_mc_estimate"],
        width=width,
        label="Plain MC",
    )
    ax.bar(
        [idx + width / 2 for idx in x],
        results_df["control_variate_estimate"],
        width=width,
        label="Control variate MC",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["greek"].str.title())
    ax.set_ylabel("Greek value")
    ax.set_title("Arithmetic Asian Greeks")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save_if_requested(fig, output_path)
    return fig

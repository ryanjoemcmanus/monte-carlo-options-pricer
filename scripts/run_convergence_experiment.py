"""Run and save the convergence experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mc_options.experiments import run_convergence_experiment
from mc_options.plotting import plot_convergence, plot_error_vs_paths


def main() -> None:
    output_tables = Path("outputs/tables")
    output_figures = Path("outputs/figures")
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    results = run_convergence_experiment(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        option_type="call",
        path_counts=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
        n_trials=10,
        seed=42,
    )
    results.to_csv(output_tables / "convergence_results.csv", index=False)

    fig = plot_convergence(results, output_figures / "convergence_price.png")
    plt.close(fig)
    fig = plot_error_vs_paths(results, output_figures / "convergence_error.png")
    plt.close(fig)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

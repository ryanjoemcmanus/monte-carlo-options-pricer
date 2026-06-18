"""Generate tables and figures used by the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mc_options.black_scholes import black_scholes_greeks, black_scholes_price
from mc_options.experiments import (
    run_asian_greeks_comparison,
    run_asian_option_experiment,
    run_convergence_experiment,
    run_greeks_comparison,
    run_moneyness_experiment,
    run_variance_reduction_experiment,
)
from mc_options.monte_carlo import (
    estimate_greeks_mc_finite_difference,
    price_european_option_mc,
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
    price_european_option_mc_sobol,
)
from mc_options.plotting import (
    plot_asian_greeks_comparison,
    plot_asian_option_methods,
    plot_convergence,
    plot_error_vs_paths,
    plot_greeks_comparison,
    plot_moneyness_comparison,
    plot_variance_reduction_comparison,
)


def _pricing_example_table() -> pd.DataFrame:
    rows = []
    methods = {
        "plain": price_european_option_mc,
        "sobol_quasi_mc": price_european_option_mc_sobol,
    }
    for option_type in ["call", "put"]:
        bs_price = black_scholes_price(100.0, 100.0, 0.05, 0.20, 1.0, option_type)
        for method_name, method_func in methods.items():
            mc_result = method_func(
                100.0, 100.0, 0.05, 0.20, 1.0, 100_000, option_type, 42
            )
            rows.append(
                {
                    "option_type": option_type,
                    "method": method_name,
                    "mc_price": mc_result["mc_price"],
                    "black_scholes_price": bs_price,
                    "standard_error": mc_result["standard_error"],
                    "ci_lower": mc_result["ci_lower"],
                    "ci_upper": mc_result["ci_upper"],
                    "absolute_error": abs(mc_result["mc_price"] - bs_price),
                    "relative_error_pct": abs(mc_result["mc_price"] - bs_price) / bs_price * 100,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pricing_examples = _pricing_example_table()
    pricing_examples.to_csv(tables_dir / "pricing_examples.csv", index=False)

    convergence = run_convergence_experiment(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
        "call",
        [1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
        n_trials=10,
        seed=42,
    )
    convergence.to_csv(tables_dir / "convergence_results.csv", index=False)

    moneyness = run_moneyness_experiment(
        S0_values=[70, 80, 90, 100, 110, 120, 130],
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        n_paths=100_000,
        option_type="call",
        seed=42,
    )
    moneyness.to_csv(tables_dir / "moneyness_comparison.csv", index=False)

    variance_reduction = run_variance_reduction_experiment(
        100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=100_000, n_trials=20, seed=42
    )
    variance_reduction.to_csv(tables_dir / "variance_reduction_results.csv", index=False)

    greeks = run_greeks_comparison(
        100.0, 100.0, 0.05, 0.20, 1.0, "call", n_paths=500_000, seed=42
    )
    greeks.to_csv(tables_dir / "greeks_comparison.csv", index=False)

    asian_options = run_asian_option_experiment(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
        "call",
        n_paths=100_000,
        n_steps=252,
        n_trials=20,
        seed=42,
    )
    asian_options.to_csv(tables_dir / "asian_option_results.csv", index=False)

    asian_greeks = run_asian_greeks_comparison(
        100.0,
        100.0,
        0.05,
        0.20,
        1.0,
        "call",
        n_paths=100_000,
        n_steps=252,
        seed=42,
    )
    asian_greeks.to_csv(tables_dir / "asian_greeks_comparison.csv", index=False)

    fig = plot_convergence(convergence, figures_dir / "convergence_price.png")
    plt.close(fig)
    fig = plot_error_vs_paths(convergence, figures_dir / "convergence_error.png")
    plt.close(fig)
    fig = plot_moneyness_comparison(moneyness, figures_dir / "moneyness_comparison.png")
    plt.close(fig)
    fig = plot_variance_reduction_comparison(
        variance_reduction, figures_dir / "variance_reduction_comparison.png"
    )
    plt.close(fig)
    fig = plot_greeks_comparison(greeks, figures_dir / "greeks_comparison.png")
    plt.close(fig)
    fig = plot_asian_option_methods(asian_options, figures_dir / "asian_option_methods.png")
    plt.close(fig)
    fig = plot_asian_greeks_comparison(
        asian_greeks, figures_dir / "asian_greeks_comparison.png"
    )
    plt.close(fig)

    print("Generated report outputs:")
    print(f"- {tables_dir / 'pricing_examples.csv'}")
    print(f"- {tables_dir / 'convergence_results.csv'}")
    print(f"- {tables_dir / 'moneyness_comparison.csv'}")
    print(f"- {tables_dir / 'variance_reduction_results.csv'}")
    print(f"- {tables_dir / 'greeks_comparison.csv'}")
    print(f"- {tables_dir / 'asian_option_results.csv'}")
    print(f"- {tables_dir / 'asian_greeks_comparison.csv'}")
    print(f"Call BS Greeks: {black_scholes_greeks(100.0, 100.0, 0.05, 0.20, 1.0, 'call')}")
    print(
        "Call MC Greeks: "
        f"{estimate_greeks_mc_finite_difference(100.0, 100.0, 0.05, 0.20, 1.0, 100_000, 'call', 42)}"
    )
    print(
        "Variance reduction sample prices: "
        f"plain={price_european_option_mc(100, 100, 0.05, 0.2, 1, 100_000, 'call', 42)['mc_price']:.4f}, "
        f"antithetic={price_european_option_mc_antithetic(100, 100, 0.05, 0.2, 1, 100_000, 'call', 42)['mc_price']:.4f}, "
        f"control={price_european_option_mc_control_variate(100, 100, 0.05, 0.2, 1, 100_000, 'call', 42)['mc_price']:.4f}, "
        f"sobol={price_european_option_mc_sobol(100, 100, 0.05, 0.2, 1, 100_000, 'call', 42)['mc_price']:.4f}"
    )


if __name__ == "__main__":
    main()

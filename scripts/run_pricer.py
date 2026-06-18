"""Run one call and put pricing example."""

from __future__ import annotations

from mc_options.black_scholes import black_scholes_price
from mc_options.monte_carlo import price_european_option_mc, price_european_option_mc_sobol


def print_result(option_type: str) -> None:
    params = {
        "S0": 100.0,
        "K": 100.0,
        "r": 0.05,
        "sigma": 0.20,
        "T": 1.0,
        "n_paths": 100_000,
        "option_type": option_type,
        "seed": 42,
    }
    mc_result = price_european_option_mc(**params)
    sobol_result = price_european_option_mc_sobol(**params)
    bs_price = black_scholes_price(
        params["S0"], params["K"], params["r"], params["sigma"], params["T"], option_type
    )
    absolute_error = abs(mc_result["mc_price"] - bs_price)
    relative_error = absolute_error / bs_price * 100

    print(f"\nEuropean {option_type.title()} Option")
    print(f"Monte Carlo price:     {mc_result['mc_price']:.6f}")
    print(f"Black-Scholes price:   {bs_price:.6f}")
    print(f"Standard error:        {mc_result['standard_error']:.6f}")
    print(f"95% confidence int.:   [{mc_result['ci_lower']:.6f}, {mc_result['ci_upper']:.6f}]")
    print(f"Absolute error:        {absolute_error:.6f}")
    print(f"Relative error:        {relative_error:.4f}%")
    print(f"Sobol QMC price:       {sobol_result['mc_price']:.6f}")
    print(f"Sobol absolute error:  {abs(sobol_result['mc_price'] - bs_price):.6f}")


if __name__ == "__main__":
    print_result("call")
    print_result("put")

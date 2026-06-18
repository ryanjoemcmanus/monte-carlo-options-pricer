"""Run arithmetic Asian option pricing examples."""

from __future__ import annotations

from mc_options.asian_options import (
    estimate_asian_greeks_mc_finite_difference,
    geometric_asian_option_price,
    price_arithmetic_asian_option_mc,
    price_arithmetic_asian_option_mc_control_variate,
)


def main() -> None:
    params = {
        "S0": 100.0,
        "K": 100.0,
        "r": 0.05,
        "sigma": 0.20,
        "T": 1.0,
        "n_paths": 100_000,
        "n_steps": 252,
        "option_type": "call",
        "seed": 42,
    }
    plain = price_arithmetic_asian_option_mc(**params)
    control = price_arithmetic_asian_option_mc_control_variate(**params)
    geometric_price = geometric_asian_option_price(
        params["S0"],
        params["K"],
        params["r"],
        params["sigma"],
        params["T"],
        params["n_steps"],
        params["option_type"],
    )

    print("Arithmetic Asian Call Option")
    print(f"Plain MC price:          {plain['mc_price']:.6f}")
    print(f"Plain standard error:    {plain['standard_error']:.6f}")
    print(f"Control-variate price:   {control['mc_price']:.6f}")
    print(f"Control standard error:  {control['standard_error']:.6f}")
    print(f"Geometric Asian price:   {geometric_price:.6f}")
    print(f"Control beta:            {control['beta']:.6f}")
    print(f"Payoff correlation:      {control['correlation']:.6f}")

    greeks = estimate_asian_greeks_mc_finite_difference(
        **params, use_control_variate=True
    )
    print("\nArithmetic Asian Greeks (finite-difference MC, control variate)")
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        print(f"{greek.title():<7} {greeks[greek]: .6f}")


if __name__ == "__main__":
    main()

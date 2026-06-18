# Monte Carlo Options Pricer

## Overview

`monte-carlo-options-pricer` is a Python derivatives analytics project that prices European and Asian options, validates Monte Carlo estimates against closed-form benchmarks, and connects the models to real listed option chains through an interactive Streamlit dashboard.

The project combines stochastic simulation, Black-Scholes pricing, variance reduction, Greeks estimation, implied volatility solving, volatility smile visualization, and historical-versus-implied volatility comparison in one cohesive analytics workflow.

## Why It Matters

Options pricing is a practical setting for several skills used in quantitative trading and derivatives research. The project demonstrates risk-neutral valuation, simulation under geometric Brownian motion, model validation, uncertainty quantification, numerical convergence, volatility analysis, and Python-based market data analysis.

The dashboard makes the workflow easy to explore interactively. Users can switch between model-driven pricing and real option-chain analysis, then see how the pricing engine behaves under different assumptions.

## Dashboard

The Streamlit dashboard has two modes.

`Model Lab` is an interactive pricing workspace for European vanilla options and arithmetic Asian options. It lets the user change spot, strike, volatility, interest rate, maturity, option type, simulation size, and pricing method. It reports Monte Carlo prices, benchmark prices, standard errors, confidence intervals, Greeks, and variance-reduction comparisons.

`Market Option Chain` connects the project to real listed options. It fetches option chains for tickers such as `SPY`, `AAPL`, `NVDA`, and `TSLA` using `yfinance`. For the selected expiration and strike, it computes the bid/ask midpoint, solves Black-Scholes implied volatility, compares the market midpoint against a flat-volatility Black-Scholes model, plots the volatility smile across strikes, and compares selected-contract implied volatility against 20-day, 60-day, and 252-day realized historical volatility.

Market data is sourced from Yahoo Finance through `yfinance`, so it should be treated as delayed demonstration data rather than production trading data.

Launch the dashboard with:

```bash
streamlit run streamlit_app.py
```

## Mathematical Model

The core Monte Carlo engine simulates terminal stock prices under risk-neutral geometric Brownian motion:

```text
S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

where `Z` is a standard normal random variable. European call and put payoffs are discounted under the risk-neutral measure and compared against the Black-Scholes closed-form solution.

The Asian option extension simulates full price paths instead of only terminal prices. A fixed-strike arithmetic Asian option pays based on the arithmetic average of monitored prices:

```text
A = (1 / n) * sum(S_t_i)
call payoff = max(A - K, 0)
put payoff = max(K - A, 0)
```

Arithmetic Asian options do not have the same simple closed-form solution as European options, which makes them a natural use case for Monte Carlo simulation. The project also implements a closed-form discretely monitored geometric Asian option price and uses the geometric Asian payoff as a control variate to reduce simulation error.

## Implementation

The code is organized as a small installable Python package under `src/mc_options`. Pricing logic, Black-Scholes formulas, Asian option logic, experiments, plotting, market-data helpers, and validation utilities are separated into focused modules. Scripts under `scripts/` generate reproducible outputs for examples, convergence experiments, variance-reduction comparisons, Greeks comparisons, and report artifacts. Tests under `tests/` cover pricing correctness, put-call parity, reproducibility, variance reduction, implied volatility inversion, Asian option behavior, market-chain enrichment, and historical volatility calculations.

The repository structure is:

```text
monte-carlo-options-pricer/
|-- README.md
|-- streamlit_app.py
|-- requirements.txt
|-- pyproject.toml
|-- src/
|   `-- mc_options/
|       |-- black_scholes.py
|       |-- monte_carlo.py
|       |-- asian_options.py
|       |-- implied_volatility.py
|       |-- market_data.py
|       |-- experiments.py
|       |-- plotting.py
|       `-- utils.py
|-- scripts/
|-- tests/
|-- outputs/
`-- notebooks/
```

## Installation

Install the project from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the test suite with:

```bash
pytest
```

Generate the report outputs with:

```bash
python scripts/generate_report_outputs.py
```

## Results

For a one-year at-the-money European call with `S0 = 100`, `K = 100`, `r = 5%`, `sigma = 20%`, and `100,000` Monte Carlo paths, the project estimated a Monte Carlo price of approximately `10.42` versus a Black-Scholes benchmark of approximately `10.45`. The benchmark was inside the reported 95% confidence interval.

The variance-reduction experiments showed that control variates materially reduce simulation error. In the European call example, the control-variate standard error was much lower than plain Monte Carlo. In the Asian option example, the geometric Asian control variate reduced standard error from roughly `0.0254` to roughly `0.0007`, which is a strong demonstration of why variance reduction matters in quantitative finance.

The Greeks comparison showed Monte Carlo finite-difference estimates close to Black-Scholes values for European options, while the Asian option module estimates risk sensitivities using full-path finite differences and common random numbers.

The market dashboard connects the pricing engine to real listed options. It solves implied volatility from observed bid/ask midpoints, plots the volatility smile across strikes, and compares selected-contract implied volatility against realized historical volatility. This gives the project a practical analytics layer beyond textbook pricing.

## Limitations

This project is intended as a quantitative finance learning and demonstration tool. It assumes constant volatility and a constant risk-free rate in the pricing models. It uses lognormal stock dynamics, does not model transaction costs or liquidity, and does not attempt production-grade market data handling. The market-data dashboard depends on Yahoo Finance availability and should not be used as a trading system.

## Future Work

Natural extensions would include barrier options, quasi-Monte Carlo sampling, an implied volatility term-structure view, a Heston stochastic volatility model, calibration to option-chain data, and more robust institutional market data integration.

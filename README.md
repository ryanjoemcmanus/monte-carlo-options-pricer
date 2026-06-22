# Monte Carlo Options Pricer

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-33%20passed-2ea44f)
![Project type](https://img.shields.io/badge/project%20type-quantitative%20finance-6f42c1)

A Python derivatives-analytics platform for Monte Carlo and Sobol quasi-Monte Carlo option pricing, Black-Scholes validation, variance reduction, Greeks estimation, implied-volatility analysis, and interactive listed-option exploration.

All numerical results below are reproducible from fixed-seed scripts in this repository. The market-data dashboard uses Yahoo Finance through `yfinance` for delayed demonstration data and is not production trading software.

## Why this matters for quant research

This project demonstrates stochastic simulation for derivatives pricing, variance reduction, and benchmark validation against Black-Scholes. It also emphasizes model-error analysis and reproducible numerical research through fixed seeds, confidence intervals, and controlled experiments.

## Highlights

- Prices European calls and puts using plain Monte Carlo, antithetic variates, control variates, and scrambled Sobol quasi-Monte Carlo.
- Prices fixed-strike arithmetic Asian options using full-path geometric Brownian motion simulation.
- Validates European option estimates against Black-Scholes closed-form prices and confidence intervals.
- Reduced the fixed-seed European call pricing error from `0.0300` with plain Monte Carlo to `0.00003` with Sobol quasi-Monte Carlo at `100,000` paths.
- Reduced 20-trial European call pricing error to `0.000024` with Sobol quasi-Monte Carlo, compared with `0.00245` for plain Monte Carlo and `0.000565` for the control variate.
- Reduced European call standard error from `0.0466` to `0.0177` using a terminal-stock-price control variate.
- Reduced arithmetic Asian option standard error from approximately `0.0254` to approximately `0.0007` using a geometric Asian control variate.
- Estimates Delta, Gamma, Vega, Theta, and Rho with finite differences and common random numbers.
- Solves implied volatility from listed-option bid/ask midpoints.
- Visualizes volatility smiles and compares implied volatility with realized historical volatility.
- Includes an interactive Streamlit dashboard for model experiments and option-chain analysis.

## Dashboard Preview

![Monte Carlo Options Pricer dashboard](docs/images/dashboard.png)

![Option-chain volatility analysis](docs/images/option_chain_dashboard.png)

## Benchmark Example

For a one-year at-the-money European call with:

- `S0 = 100`
- `K = 100`
- `r = 5%`
- `sigma = 20%`
- `100,000` simulated paths
- `seed = 42`

| Method | Price | Absolute Error | Relative Difference |
|---|---:|---:|---:|
| Plain Monte Carlo | `10.4205` | `0.0300` | `0.2875%` |
| Sobol quasi-Monte Carlo | `10.4506` | `0.00003` | `0.0003%` |
| Black-Scholes | `10.4506` | `0.0000` | `0.0000%` |

The Black-Scholes benchmark falls inside the plain Monte Carlo 95% confidence interval `[10.3289, 10.5122]`.

## Variance Reduction and Quasi-Monte Carlo

| Experiment | Baseline | Improved Result | Method |
|---|---:|---:|---|
| European call standard error | `0.0466` | `0.0177` | Terminal stock-price control variate |
| European call standard error | `0.0466` | `0.0328` | Antithetic variates |
| European call absolute pricing error | `0.00245` | `0.000024` | Sobol quasi-Monte Carlo |
| Arithmetic Asian call standard error | `0.0254` | `0.0007` | Geometric Asian control variate |

The arithmetic Asian result uses the geometric Asian payoff as a highly correlated control variate while preserving the arithmetic Asian option as the pricing target. In the fixed-seed single-run example, payoff correlation was `0.9996` and control beta was `1.0358`.

## Overview

This project is a compact derivatives analytics workflow. It starts with risk-neutral simulation under geometric Brownian motion, validates European option prices against Black-Scholes, adds variance reduction and Sobol quasi-Monte Carlo, estimates Greeks, prices path-dependent arithmetic Asian options, and connects the analytics to a Streamlit dashboard for listed-option exploration.

The core research lesson is that Monte Carlo pricing is not just about producing a point estimate. A useful pricing engine should quantify uncertainty, validate against known benchmarks, reduce estimator variance when possible, and expose model assumptions clearly.

## Dashboard Modes

`Model Lab` prices European vanilla and arithmetic Asian options interactively. Users can change spot, strike, volatility, interest rate, maturity, option type, simulation count, random seed, and pricing method. European methods include plain Monte Carlo, antithetic variates, a terminal-stock-price control variate, and scrambled Sobol quasi-Monte Carlo. The dashboard reports price, benchmark price, standard error, confidence interval, relative error, Greeks, and method comparisons.

`Market Option Chain` fetches listed-option chains with `yfinance`, computes bid/ask midpoints, solves Black-Scholes implied volatility, compares market midpoint against a flat-volatility Black-Scholes model, plots the implied-volatility smile, and compares selected-contract implied volatility with 20-day, 60-day, and 252-day realized historical volatility.

Yahoo Finance data should be treated as delayed demonstration data. The dashboard is designed for research presentation and exploration, not execution or production market-data use.

## Mathematical Model

The European Monte Carlo engine simulates terminal prices under risk-neutral geometric Brownian motion:

$$
S_T = S_0 \exp\left[\left(r-\frac{1}{2}\sigma^2\right)T
+\sigma\sqrt{T}Z\right],
\qquad Z\sim\mathcal{N}(0,1).
$$

The Sobol quasi-Monte Carlo engine replaces pseudo-random normal draws with one-dimensional scrambled Sobol points transformed through the inverse normal CDF. Repeated Sobol trials use different scrambles, which makes pricing-error comparisons reproducible while preserving the low-discrepancy structure.

European call and put payoffs are:

$$
\max(S_T-K,0)
\qquad\text{and}\qquad
\max(K-S_T,0).
$$

The arithmetic Asian extension simulates full paths and prices options on the monitored arithmetic average:

$$
A = \frac{1}{n}\sum_{i=1}^{n} S_{t_i}.
$$

The fixed-strike Asian payoffs are:

$$
\max(A-K,0)
\qquad\text{and}\qquad
\max(K-A,0).
$$

Arithmetic Asian options do not have the same simple closed-form benchmark as European options, so the project also implements a discretely monitored geometric Asian closed-form price and uses the simulated geometric payoff as a control variate.

## Implementation

The code is organized as an installable package under `src/mc_options`. Core modules separate Black-Scholes formulas, Monte Carlo and Sobol pricing, Asian option simulation, implied-volatility inversion, market-data enrichment, experiment generation, plotting, and validation helpers.

Monte Carlo Greeks use central finite differences with common random numbers. This keeps the random shocks aligned across bumped scenarios and reduces finite-difference noise.

## Numerical Results

### Monte Carlo Convergence

The convergence experiment compares mean plain Monte Carlo and scrambled Sobol quasi-Monte Carlo call prices against the Black-Scholes benchmark across path counts from `1,000` to `500,000`, with 10 trials per path count.

![Monte Carlo convergence](outputs/figures/convergence_price.png)

At `100,000` paths, the 10-trial plain Monte Carlo mean price was `10.4468`, a relative error of `0.0363%`. The Sobol mean price was `10.4505`, reducing relative error to `0.0004%`. At `500,000` paths, Sobol relative error was `0.0001%`.

### European Variance Reduction and Sobol QMC

The European method comparison evaluates plain Monte Carlo, antithetic variates, a terminal-stock-price control variate, and scrambled Sobol quasi-Monte Carlo over 20 trials.

![European variance-reduction comparison](outputs/figures/variance_reduction_comparison.png)

| Method | Mean Price | Mean Std. Error | Absolute Error vs Black-Scholes |
|---|---:|---:|---:|
| Plain Monte Carlo | `10.4481` | `0.0466` | `0.0025` |
| Antithetic variates | `10.4450` | `0.0328` | `0.0056` |
| Control variate | `10.4511` | `0.0177` | `0.0006` |
| Sobol quasi-Monte Carlo | `10.4506` | `0.0465` | `0.000024` |

For Sobol quasi-Monte Carlo, the payoff-sample standard error is shown for comparability, but the stronger diagnostic is pricing error across independent scrambles. The 20-trial Sobol price standard deviation was `0.000239`, compared with `0.0525` for plain Monte Carlo.

### Asian Option Variance Reduction

The Asian experiment prices a one-year fixed-strike arithmetic Asian call with 252 monitoring dates and `100,000` paths.

![Arithmetic Asian option methods](outputs/figures/asian_option_methods.png)

| Method | Mean Price | Mean Std. Error | Geometric Asian Benchmark |
|---|---:|---:|---:|
| Plain arithmetic Asian MC | `5.7906` | `0.0254` | `5.5655` |
| Geometric control variate | `5.7820` | `0.0007` | `5.5655` |

### Greeks

For a one-year at-the-money European call with `500,000` paths, finite-difference Monte Carlo Greeks were close to the Black-Scholes values.

![Greeks comparison](outputs/figures/greeks_comparison.png)

| Greek | Black-Scholes | Monte Carlo | Relative Error |
|---|---:|---:|---:|
| Delta | `0.6368` | `0.6361` | `0.1144%` |
| Gamma | `0.0188` | `0.0188` | `0.0871%` |
| Vega | `37.5240` | `37.5621` | `0.1016%` |
| Theta | `-6.4140` | `-6.4143` | `0.0039%` |
| Rho | `53.2325` | `53.1599` | `0.1363%` |

## Installation

Install dependencies and the local package from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Run the core examples:

```bash
python scripts/run_pricer.py
python scripts/run_asian_options.py
```

Regenerate all report tables and figures:

```bash
python scripts/generate_report_outputs.py
```

Outputs are written to `outputs/tables/` and `outputs/figures/`.

## Validation and Testing

Run the full test suite:

```bash
pytest
```

The suite currently contains 33 tests. It covers fixed-seed Monte Carlo reproducibility, Sobol terminal-price reproducibility, Black-Scholes prices and Greeks, invalid input handling, put-call parity, confidence-interval behavior, antithetic variates, control variates, Sobol quasi-Monte Carlo pricing, finite-difference Greeks, Asian path simulation, geometric Asian pricing, Asian control-variate variance reduction, Asian Greeks, implied-volatility inversion, option-chain enrichment, and historical-volatility calculations.

Latest local verification:

```text
33 passed
```

## Repository Structure

```text
monte-carlo-options-pricer/
|-- README.md
|-- streamlit_app.py
|-- requirements.txt
|-- pyproject.toml
|-- docs/
|   `-- images/
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

## Limitations

The pricing models assume geometric Brownian motion, constant volatility, constant interest rates, no dividends, no transaction costs, no market impact, and no liquidity constraints. The Streamlit market-data workflow depends on Yahoo Finance availability through `yfinance`; it is delayed demonstration data and does not provide production-grade market-data guarantees.

The dashboard and scripts are intended for derivatives analytics, numerical-method demonstration, and project review. They are not execution systems and should not be used as trading infrastructure.

## Future Work

Natural extensions include barrier options, Brownian-bridge Sobol paths for high-dimensional path-dependent products, implied-volatility term-structure views, Heston stochastic-volatility pricing, dividend-yield support, calibration to option-chain data, local-volatility surfaces, and more robust institutional market-data integrations.

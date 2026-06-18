"""Interactive Streamlit dashboard for the options pricing project."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc_options.asian_options import (
    estimate_asian_greeks_mc_finite_difference,
    geometric_asian_option_price,
    price_arithmetic_asian_option_mc,
    price_arithmetic_asian_option_mc_control_variate,
)
from mc_options.black_scholes import black_scholes_greeks, black_scholes_price
from mc_options.experiments import run_asian_option_experiment, run_variance_reduction_experiment
from mc_options.market_data import (
    enrich_option_chain,
    fetch_yfinance_close_prices,
    fetch_yfinance_option_chain,
    fetch_yfinance_option_snapshot,
    realized_volatility_summary,
    years_to_expiration,
)
from mc_options.monte_carlo import (
    estimate_greeks_mc_finite_difference,
    price_european_option_mc,
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
)


st.set_page_config(
    page_title="Monte Carlo Options Pricer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown(
    """
    <style>
    :root {
        --panel-border: rgba(68, 76, 86, 0.18);
        --muted: #65758b;
        --accent: #0f766e;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid var(--panel-border);
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 760;
        line-height: 1.35;
        margin: 0 0 0.2rem 0;
        padding-top: 0.1rem;
        color: #111827;
    }
    .app-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 1rem;
    }
    .section-label {
        color: #0f766e;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .metric-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: -0.5rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--panel-border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 104px;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
    }
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-size: 1.45rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--panel-border);
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.5rem;
        border-radius: 6px 6px 0 0;
    }
    .stDataFrame {
        border: 1px solid var(--panel-border);
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_money(value: float) -> str:
    return f"${value:,.4f}"


def _format_number(value: float) -> str:
    return f"{value:,.6f}"


def _relative_error(estimate: float, benchmark: float) -> float:
    if benchmark == 0:
        return math.nan
    return abs(estimate - benchmark) / abs(benchmark) * 100


@st.cache_data(show_spinner=False)
def _price_european_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    if method == "Plain Monte Carlo":
        return price_european_option_mc(S0, K, r, sigma, T, n_paths, option_type, seed)
    if method == "Antithetic Variates":
        even_paths = n_paths if n_paths % 2 == 0 else n_paths + 1
        return price_european_option_mc_antithetic(
            S0, K, r, sigma, T, even_paths, option_type, seed
        )
    return price_european_option_mc_control_variate(S0, K, r, sigma, T, n_paths, option_type, seed)


@st.cache_data(show_spinner=False)
def _price_asian_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    option_type: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    if method == "Plain Monte Carlo":
        return price_arithmetic_asian_option_mc(
            S0, K, r, sigma, T, n_paths, n_steps, option_type, seed
        )
    return price_arithmetic_asian_option_mc_control_variate(
        S0, K, r, sigma, T, n_paths, n_steps, option_type, seed
    )


@st.cache_data(show_spinner=False)
def _european_greeks_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str,
    seed: int,
) -> tuple[dict[str, float], dict[str, float | int | str]]:
    return (
        black_scholes_greeks(S0, K, r, sigma, T, option_type),
        estimate_greeks_mc_finite_difference(S0, K, r, sigma, T, n_paths, option_type, seed),
    )


@st.cache_data(show_spinner=False)
def _asian_greeks_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    option_type: str,
    seed: int,
) -> dict[str, float | int | str | bool]:
    return estimate_asian_greeks_mc_finite_difference(
        S0, K, r, sigma, T, n_paths, n_steps, option_type, seed, use_control_variate=True
    )


@st.cache_data(show_spinner=False)
def _variance_reduction_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int,
    seed: int,
) -> pd.DataFrame:
    return run_variance_reduction_experiment(
        S0, K, r, sigma, T, option_type, n_paths=n_paths, n_trials=8, seed=seed
    )


@st.cache_data(show_spinner=False)
def _asian_methods_cached(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> pd.DataFrame:
    return run_asian_option_experiment(
        S0,
        K,
        r,
        sigma,
        T,
        option_type,
        n_paths=n_paths,
        n_steps=n_steps,
        n_trials=8,
        seed=seed,
    )


def _greeks_table(values: dict[str, float | int | str | bool], label: str) -> pd.DataFrame:
    rows = []
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        rows.append({"greek": greek.title(), label: float(values[greek])})
    return pd.DataFrame(rows)


def _plot_bar(df: pd.DataFrame, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", ax=ax, width=0.72)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.xticks(rotation=0)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data(show_spinner=False, ttl=900)
def _market_snapshot_cached(ticker: str) -> tuple[float, list[str]]:
    return fetch_yfinance_option_snapshot(ticker)


@st.cache_data(show_spinner=False, ttl=900)
def _market_chain_cached(
    ticker: str,
    expiration: str,
    option_type: str,
    spot: float,
    r: float,
    flat_sigma: float,
) -> tuple[pd.DataFrame, float]:
    T = years_to_expiration(expiration)
    raw_chain = fetch_yfinance_option_chain(ticker, expiration, option_type)
    enriched = enrich_option_chain(raw_chain, spot, r, T, option_type, flat_sigma)
    return enriched, T


@st.cache_data(show_spinner=False, ttl=900)
def _historical_volatility_cached(ticker: str) -> pd.DataFrame:
    close_prices = fetch_yfinance_close_prices(ticker)
    return realized_volatility_summary(close_prices)


def _plot_smile(chain_df: pd.DataFrame, selected_strike: float) -> None:
    smile_df = chain_df.dropna(subset=["solved_iv"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if not smile_df.empty:
        ax.plot(smile_df["strike"], smile_df["solved_iv"] * 100, marker="o", linewidth=1.8)
        ax.axvline(selected_strike, color="#0f766e", linestyle="--", linewidth=1.4)
    ax.set_title("Implied Volatility Smile")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Solved implied volatility (%)")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _plot_market_vs_model(chain_df: pd.DataFrame, selected_strike: float) -> None:
    plot_df = chain_df.dropna(subset=["mid_price", "flat_vol_model_price"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if not plot_df.empty:
        ax.plot(plot_df["strike"], plot_df["mid_price"], marker="o", label="Market midpoint")
        ax.plot(
            plot_df["strike"],
            plot_df["flat_vol_model_price"],
            marker="s",
            linestyle="--",
            label="Flat-vol Black-Scholes",
        )
        ax.axvline(selected_strike, color="#0f766e", linestyle="--", linewidth=1.4)
    ax.set_title("Market Price vs Flat-Vol Model")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option price")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _plot_iv_hv_comparison(hv_df: pd.DataFrame, selected_iv: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot_df = hv_df.copy()
    labels = plot_df["window"].tolist() + ["Selected IV"]
    values = (plot_df["historical_volatility"].tolist() + [selected_iv])
    ax.bar(labels, [value * 100 if pd.notna(value) else np.nan for value in values], color="#0f766e")
    ax.set_title("Implied Volatility vs Historical Volatility")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _default_expiration_index(expirations: list[str], target_days: int = 30) -> int:
    year_fracs = [years_to_expiration(expiry) for expiry in expirations]
    target_years = target_days / 365
    return min(range(len(expirations)), key=lambda idx: abs(year_fracs[idx] - target_years))


def _render_market_dashboard() -> None:
    with st.sidebar:
        st.markdown("### Market Option Chain")
        ticker = st.text_input("Ticker", value="SPY").strip().upper()
        market_option_type = st.segmented_control("Option type", ["call", "put"], default="call")
        market_r_pct = st.slider("Risk-free rate", 0.0, 12.0, 5.0, 0.25, key="market_r")
        flat_vol_pct = st.slider("Flat model volatility", 1.0, 120.0, 20.0, 0.5)
        moneyness_min, moneyness_max = st.slider(
            "Strike / spot range",
            0.5,
            1.5,
            (0.8, 1.2),
            0.01,
        )

    if not ticker:
        st.info("Enter a ticker to load its listed options.")
        return

    try:
        spot, expirations = _market_snapshot_cached(ticker)
    except Exception as exc:
        st.error(f"Could not load market data for {ticker}: {exc}")
        return

    expiry = st.selectbox(
        "Expiration",
        expirations,
        index=_default_expiration_index(expirations),
    )
    r = market_r_pct / 100
    flat_sigma = flat_vol_pct / 100

    try:
        chain_df, T = _market_chain_cached(ticker, expiry, market_option_type, spot, r, flat_sigma)
    except Exception as exc:
        st.error(f"Could not load option chain for {ticker} {expiry}: {exc}")
        return

    filtered_df = chain_df[
        (chain_df["moneyness"] >= moneyness_min) & (chain_df["moneyness"] <= moneyness_max)
    ].copy()
    if filtered_df.empty:
        st.warning("No contracts matched the selected strike range.")
        return

    default_idx = int((filtered_df["strike"] - spot).abs().idxmin())
    selected_strike = st.selectbox(
        "Contract strike",
        filtered_df["strike"].tolist(),
        index=filtered_df.index.tolist().index(default_idx),
    )
    selected = filtered_df.loc[filtered_df["strike"] == selected_strike].iloc[0]
    try:
        hv_df = _historical_volatility_cached(ticker)
    except Exception as exc:
        hv_df = pd.DataFrame(
            columns=["window", "trading_days", "historical_volatility"]
        )
        st.warning(f"Could not load historical volatility for {ticker}: {exc}")

    selected_iv = float(selected["solved_iv"]) if pd.notna(selected["solved_iv"]) else np.nan
    hv_20 = hv_df.loc[hv_df["window"] == "20d", "historical_volatility"]
    hv_20_value = float(hv_20.iloc[0]) if not hv_20.empty and pd.notna(hv_20.iloc[0]) else np.nan
    iv_hv_spread = selected_iv - hv_20_value if pd.notna(selected_iv) and pd.notna(hv_20_value) else np.nan
    iv_hv_ratio = selected_iv / hv_20_value if pd.notna(iv_hv_spread) and hv_20_value > 0 else np.nan

    cols = st.columns(5)
    cols[0].metric("Spot", _format_money(spot))
    cols[1].metric("Market midpoint", _format_money(float(selected["mid_price"])))
    cols[2].metric("Solved IV", f"{float(selected['solved_iv']) * 100:.2f}%" if pd.notna(selected["solved_iv"]) else "N/A")
    cols[3].metric("Flat-vol model", _format_money(float(selected["flat_vol_model_price"])))
    cols[4].metric("Model - market", _format_money(float(selected["model_minus_mid"])))

    st.markdown(
        f'<div class="metric-note">Ticker: {ticker} | Expiration: {expiry} | T: {T:.4f} years | Option type: {market_option_type}</div>',
        unsafe_allow_html=True,
    )

    hv_cols = st.columns(4)
    hv_cols[0].metric("20d historical vol", f"{hv_20_value * 100:.2f}%" if pd.notna(hv_20_value) else "N/A")
    hv_cols[1].metric("IV - 20d HV", f"{iv_hv_spread * 100:.2f}%" if pd.notna(iv_hv_spread) else "N/A")
    hv_cols[2].metric("IV / 20d HV", f"{iv_hv_ratio:.2f}x" if pd.notna(iv_hv_ratio) else "N/A")
    hv_cols[3].metric("Flat vol input", f"{flat_vol_pct:.2f}%")

    selected_tab, smile_tab, hv_tab, chain_tab = st.tabs(
        ["Selected Contract", "Volatility Smile", "IV vs HV", "Option Chain"]
    )
    with selected_tab:
        st.markdown('<div class="section-label">Contract Analytics</div>', unsafe_allow_html=True)
        selected_df = pd.DataFrame(
            [
                ["Contract", selected["contract_symbol"]],
                ["Strike", selected["strike"]],
                ["Bid", selected["bid"]],
                ["Ask", selected["ask"]],
                ["Last price", selected["last_price"]],
                ["Midpoint", selected["mid_price"]],
                ["Yahoo IV", selected["market_iv"]],
                ["Solved IV", selected["solved_iv"]],
                ["Flat-vol model price", selected["flat_vol_model_price"]],
                ["Model minus midpoint", selected["model_minus_mid"]],
                ["Volume", selected["volume"]],
                ["Open interest", selected["open_interest"]],
            ],
            columns=["Metric", "Value"],
        )
        selected_df["Value"] = selected_df["Value"].astype(str)
        st.dataframe(selected_df, hide_index=True, width="stretch")
        _plot_market_vs_model(filtered_df, float(selected_strike))

    with smile_tab:
        st.markdown('<div class="section-label">Implied Volatility by Strike</div>', unsafe_allow_html=True)
        _plot_smile(filtered_df, float(selected_strike))

    with hv_tab:
        st.markdown('<div class="section-label">Implied vs Historical Volatility</div>', unsafe_allow_html=True)
        hv_display = hv_df.copy()
        if not hv_display.empty:
            hv_display["historical_volatility_pct"] = hv_display["historical_volatility"] * 100
            hv_display["selected_iv_pct"] = selected_iv * 100 if pd.notna(selected_iv) else np.nan
            hv_display["iv_minus_hv_pct"] = hv_display["selected_iv_pct"] - hv_display["historical_volatility_pct"]
            hv_display["iv_to_hv_ratio"] = hv_display["selected_iv_pct"] / hv_display["historical_volatility_pct"]
        st.dataframe(hv_display, hide_index=True, width="stretch")
        if not hv_df.empty and pd.notna(selected_iv):
            _plot_iv_hv_comparison(hv_df, selected_iv)

    with chain_tab:
        st.markdown('<div class="section-label">Filtered Chain</div>', unsafe_allow_html=True)
        display_cols = [
            "contract_symbol",
            "strike",
            "bid",
            "ask",
            "last_price",
            "mid_price",
            "market_iv",
            "solved_iv",
            "flat_vol_model_price",
            "model_minus_mid",
            "volume",
            "open_interest",
        ]
        st.dataframe(filtered_df[display_cols], hide_index=True, width="stretch")


st.markdown('<div class="app-title">Monte Carlo Options Pricer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">European, Asian, and listed option-chain analytics with Monte Carlo simulation, Black-Scholes validation, implied volatility, variance reduction, confidence intervals, and Greeks.</div>',
    unsafe_allow_html=True,
)

dashboard_mode = st.segmented_control(
    "Dashboard mode",
    ["Model Lab", "Market Option Chain"],
    default="Model Lab",
)

if dashboard_mode == "Market Option Chain":
    _render_market_dashboard()
    st.stop()

with st.sidebar:
    st.markdown("### Pricing Controls")
    product = st.radio("Product", ["European Vanilla", "Arithmetic Asian"], horizontal=False)
    option_type = st.segmented_control("Option type", ["call", "put"], default="call")

    if product == "European Vanilla":
        method = st.selectbox(
            "Pricing method",
            ["Plain Monte Carlo", "Antithetic Variates", "Control Variate"],
            index=2,
        )
    else:
        method = st.selectbox(
            "Pricing method",
            ["Plain Monte Carlo", "Geometric Control Variate"],
            index=1,
        )

    st.markdown("### Market Inputs")
    S0 = st.slider("Spot price S0", 40.0, 180.0, 100.0, 1.0)
    K = st.slider("Strike K", 40.0, 180.0, 100.0, 1.0)
    r_pct = st.slider("Risk-free rate", 0.0, 12.0, 5.0, 0.25)
    sigma_pct = st.slider("Volatility", 1.0, 80.0, 20.0, 0.5)
    T = st.slider("Maturity in years", 0.05, 5.0, 1.0, 0.05)

    st.markdown("### Simulation")
    n_paths = st.select_slider(
        "Monte Carlo paths",
        options=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000],
        value=100_000,
    )
    n_steps = 252
    if product == "Arithmetic Asian":
        n_steps = st.select_slider(
            "Monitoring dates",
            options=[12, 24, 52, 126, 252],
            value=252,
        )
    seed = st.number_input("Random seed", min_value=1, max_value=1_000_000, value=42, step=1)

r = r_pct / 100
sigma = sigma_pct / 100

if product == "European Vanilla":
    result = _price_european_cached(S0, K, r, sigma, T, n_paths, option_type, method, int(seed))
    benchmark = black_scholes_price(S0, K, r, sigma, T, option_type)
    benchmark_label = "Black-Scholes"
else:
    result = _price_asian_cached(
        S0, K, r, sigma, T, n_paths, n_steps, option_type, method, int(seed)
    )
    benchmark = geometric_asian_option_price(S0, K, r, sigma, T, n_steps, option_type)
    benchmark_label = "Geometric Asian"

absolute_error = abs(result["mc_price"] - benchmark)
relative_error = _relative_error(result["mc_price"], benchmark)

metric_cols = st.columns(5)
metric_cols[0].metric("Monte Carlo price", _format_money(result["mc_price"]))
metric_cols[1].metric(benchmark_label, _format_money(benchmark))
metric_cols[2].metric("Standard error", _format_number(result["standard_error"]))
metric_cols[3].metric("Absolute error", _format_number(absolute_error))
metric_cols[4].metric("Relative error", f"{relative_error:.3f}%")

ci_cols = st.columns([2, 1, 1])
ci_cols[0].markdown(
    f'<div class="metric-note">95% confidence interval: [{result["ci_lower"]:.4f}, {result["ci_upper"]:.4f}]</div>',
    unsafe_allow_html=True,
)
ci_cols[1].markdown(
    f'<div class="metric-note">Paths: {result["n_paths"]:,}</div>',
    unsafe_allow_html=True,
)
if product == "Arithmetic Asian":
    ci_cols[2].markdown(
        f'<div class="metric-note">Monitoring dates: {n_steps}</div>',
        unsafe_allow_html=True,
    )

tab_summary, tab_greeks, tab_methods = st.tabs(["Pricing Summary", "Greeks", "Method Comparison"])

with tab_summary:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="section-label">Scenario</div>', unsafe_allow_html=True)
        scenario_df = pd.DataFrame(
            [
                ["Product", product],
                ["Option type", option_type],
                ["Method", method],
                ["Spot", S0],
                ["Strike", K],
                ["Risk-free rate", f"{r_pct:.2f}%"],
                ["Volatility", f"{sigma_pct:.2f}%"],
                ["Maturity", f"{T:.2f} years"],
                ["Paths", f"{n_paths:,}"],
                ["Monitoring dates", n_steps if product == "Arithmetic Asian" else "N/A"],
            ],
            columns=["Input", "Value"],
        )
        scenario_df["Value"] = scenario_df["Value"].astype(str)
        st.dataframe(scenario_df, hide_index=True, width="stretch")
    with right:
        st.markdown('<div class="section-label">Estimator Details</div>', unsafe_allow_html=True)
        detail_rows = [
            ["MC price", result["mc_price"]],
            [benchmark_label, benchmark],
            ["Standard error", result["standard_error"]],
            ["CI lower", result["ci_lower"]],
            ["CI upper", result["ci_upper"]],
        ]
        if "beta" in result:
            detail_rows.append(["Control beta", result["beta"]])
        if "correlation" in result:
            detail_rows.append(["Payoff correlation", result["correlation"]])
        detail_df = pd.DataFrame(detail_rows, columns=["Metric", "Value"])
        detail_df["Value"] = detail_df["Value"].astype(str)
        st.dataframe(detail_df, hide_index=True, width="stretch")

with tab_greeks:
    st.markdown('<div class="section-label">Risk Sensitivities</div>', unsafe_allow_html=True)
    if product == "European Vanilla":
        bs_greeks, mc_greeks = _european_greeks_cached(
            S0, K, r, sigma, T, min(n_paths, 250_000), option_type, int(seed)
        )
        bs_df = _greeks_table(bs_greeks, "Black-Scholes")
        mc_df = _greeks_table(mc_greeks, "Finite-difference MC")
        greeks_df = bs_df.merge(mc_df, on="greek")
        greeks_df["absolute_error"] = (
            greeks_df["Finite-difference MC"] - greeks_df["Black-Scholes"]
        ).abs()
        st.dataframe(greeks_df, hide_index=True, width="stretch")
        _plot_bar(
            greeks_df.set_index("greek")[["Black-Scholes", "Finite-difference MC"]],
            "European Greeks",
            "Greek value",
        )
    else:
        asian_greeks = _asian_greeks_cached(
            S0, K, r, sigma, T, min(n_paths, 150_000), n_steps, option_type, int(seed)
        )
        greeks_df = _greeks_table(asian_greeks, "Control-variate finite-difference MC")
        st.dataframe(greeks_df, hide_index=True, width="stretch")
        _plot_bar(greeks_df.set_index("greek"), "Arithmetic Asian Greeks", "Greek value")

with tab_methods:
    st.markdown('<div class="section-label">Efficiency Comparison</div>', unsafe_allow_html=True)
    if product == "European Vanilla":
        methods_df = _variance_reduction_cached(
            S0, K, r, sigma, T, option_type, min(n_paths, 100_000), int(seed)
        )
        st.dataframe(methods_df, hide_index=True, width="stretch")
        _plot_bar(
            methods_df.set_index("method")[["mean_standard_error"]],
            "European Method Efficiency",
            "Mean standard error",
        )
    else:
        methods_df = _asian_methods_cached(
            S0, K, r, sigma, T, option_type, min(n_paths, 100_000), n_steps, int(seed)
        )
        st.dataframe(methods_df, hide_index=True, width="stretch")
        _plot_bar(
            methods_df.set_index("method")[["mean_standard_error"]],
            "Asian Method Efficiency",
            "Mean standard error",
        )

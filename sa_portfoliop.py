import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR  = os.path.join(BASE_DIR, "graphs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


TICKERS = {
    "AAPL":  "Apple Inc.",
    "AMZN":  "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "JNJ":   "Johnson & Johnson",
    "JPM":   "JPMorgan Chase & Co.",
    "MSFT":  "Microsoft Corporation",
    "NVDA":  "NVIDIA Corporation",
    "PG":    "The Procter & Gamble Company",
    "V":     "Visa Inc.",
    "XOM":   "Exxon Mobil Corporation",
}

START_DATE     = "2013-01-01"
END_DATE       = "2023-01-01"
RISK_FREE_RATE = 0.02
MAX_WEIGHT     = 0.30

SA_RUNS      = 30
SA_MAX_ITER  = 1000
SA_L         = 50
SA_T_INIT    = 1000.0
SA_T_MIN     = 1e-6
SA_COOLING_R = 0.95
SA_IP        = 0.9
SA_STEP      = 0.05

SEED = 70748647
np.random.seed(SEED)


def download_and_save():
    symbols = list(TICKERS.keys())

    print("STEP 1: Downloading data from Yahoo Finance")
    print(f"Tickers : {', '.join(symbols)}")
    print(f"Period  : {START_DATE} to {END_DATE}")

    raw = yf.download(symbols, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=True)

    for ticker in symbols:
        try:
            df_ticker = raw.xs(ticker, axis=1, level=1)
            df_ticker.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
            print(f"Saved data/{ticker}.csv ({len(df_ticker)} rows)")
        except Exception as e:
            print(f"WARNING: Could not save {ticker}: {e}")

    close_df = raw["Close"][symbols].dropna(how="all")
    close_df.to_csv(os.path.join(DATA_DIR, "close_prices_all.csv"))

    returns_df = close_df.pct_change().dropna()
    returns_df.to_csv(os.path.join(DATA_DIR, "daily_returns_all.csv"))

    print(f"Combined close prices and daily returns saved.\n")
    return close_df, returns_df


def repair_weights(weights):
    weights = np.abs(np.array(weights, dtype=float))
    weights = np.clip(weights, 0.0, MAX_WEIGHT)
    total = weights.sum()
    if total == 0:
        return np.ones(len(weights)) / len(weights)
    return weights / total


def calc_annual_return(weights, mean_ret):
    return float(np.dot(repair_weights(weights), mean_ret) * 252)


def calc_annual_risk(weights, cov_matrix):
    w = repair_weights(weights)
    return float(np.sqrt(w @ cov_matrix @ w * 252))


def calc_sharpe(weights, mean_ret, cov_matrix, rf=RISK_FREE_RATE):
    ret = calc_annual_return(weights, mean_ret)
    risk = calc_annual_risk(weights, cov_matrix)
    risk = max(risk, 0.05)
    return (ret - rf) / risk


def calc_max_drawdown(weights, close_df):
    w = repair_weights(weights)
    daily_returns = close_df.pct_change().dropna().values @ w
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def simulated_annealing(mean_ret, cov_matrix, dim, convergence_trace=None):
    current = repair_weights(np.random.uniform(0, 1, dim))
    best = current.copy()
    best_sharpe = calc_sharpe(best, mean_ret, cov_matrix)
    temperature = SA_T_INIT

    for _ in range(SA_MAX_ITER):
        accepted = 0

        for _ in range(SA_L):
            candidate = repair_weights(current + np.random.uniform(-SA_STEP, SA_STEP, dim))

            current_sharpe = calc_sharpe(current, mean_ret, cov_matrix)
            candidate_sharpe = calc_sharpe(candidate, mean_ret, cov_matrix)
            delta = candidate_sharpe - current_sharpe

            if delta >= 0 or np.random.rand() < np.exp(delta / temperature):
                current = candidate
                accepted += 1

                if candidate_sharpe > best_sharpe:
                    best = current.copy()
                    best_sharpe = candidate_sharpe

        acceptance_ratio = accepted / SA_L
        if acceptance_ratio > SA_IP:
            temperature = temperature / 2
        else:
            temperature = temperature * SA_COOLING_R

        if convergence_trace is not None:
            convergence_trace.append(best_sharpe)

        if temperature < SA_T_MIN:
            break

    if convergence_trace is not None and len(convergence_trace) < SA_MAX_ITER:
        last_value = convergence_trace[-1] if convergence_trace else best_sharpe
        convergence_trace += [last_value] * (SA_MAX_ITER - len(convergence_trace))

    return best


def run_optimizer(mean_ret, cov_matrix, dim):
    print("STEP 2: Running Simulated Annealing Optimizer")
    print(f"Stocks: {dim}, Runs: {SA_RUNS}, Iterations: {SA_MAX_ITER}, "
          f"Inner loop: {SA_L}, Seed: {SEED}\n")

    best_w = None
    best_sharpe = -np.inf
    best_run_index = -1
    all_sharpes = []
    all_traces = []
    all_weights = []
    best_trace = []

    for run in range(1, SA_RUNS + 1):
        trace = []
        weights = simulated_annealing(mean_ret, cov_matrix, dim, convergence_trace=trace)
        sharpe = calc_sharpe(weights, mean_ret, cov_matrix)

        all_sharpes.append(sharpe)
        all_traces.append(trace)
        all_weights.append(weights.copy())
        print(f"Run {run}/{SA_RUNS}: Sharpe = {sharpe:.4f}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = weights.copy()
            best_trace = trace[:]
            best_run_index = run - 1

    mean_trace = np.mean(np.array(all_traces), axis=0)

    print(f"\nBest Sharpe: {best_sharpe:.4f}\n")
    return best_w, all_sharpes, all_weights, best_trace, mean_trace, best_run_index


def build_runs_report(all_sharpes, all_weights, close_df, mean_ret, cov_matrix):
    run_returns = []
    run_risks = []
    run_sharpes = []

    sharpe_arr_full = np.array(all_sharpes, dtype=float)
    best_run_idx = int(sharpe_arr_full.argmax())

    run_rows = []
    for i, w in enumerate(all_weights, start=1):
        r = calc_annual_return(w, mean_ret)
        risk = calc_annual_risk(w, cov_matrix)
        sh = calc_sharpe(w, mean_ret, cov_matrix)

        run_returns.append(r)
        run_risks.append(risk)
        run_sharpes.append(sh)

        run_label = f"{i} (Best)" if (i - 1) == best_run_idx else str(i)

        run_rows.append([
            run_label,
            f"{r * 100:.2f}%",
            f"{risk * 100:.2f}%",
            f"{sh:.4f}",
        ])

    runs_table = tabulate(
        run_rows,
        headers=["Run", "Return", "Risk", "Sharpe Ratio"],
        tablefmt="pipe",
    )

    sharpe_arr = np.array(run_sharpes, dtype=float)
    summary_rows = [[
        "Sharpe Ratio",
        f"{sharpe_arr.mean():.4f}",
        f"{sharpe_arr.min():.4f}",
        f"{sharpe_arr.max():.4f}",
        f"{sharpe_arr.std():.4f}",
    ]]

    summary_table = tabulate(
        summary_rows,
        headers=["Metric", "Mean", "Min", "Max", "Std Dev"],
        tablefmt="pipe",
    )

    return runs_table, summary_table, run_returns, run_risks, run_sharpes


def build_report(best_w, close_df, returns_df, mean_ret, cov_matrix,
                  runs_table=None, summary_table=None):
    ticker_list = list(TICKERS.keys())
    date_start = START_DATE
    date_end = END_DATE
    trading_days = len(returns_df)

    port_return = calc_annual_return(best_w, mean_ret)
    port_risk = calc_annual_risk(best_w, cov_matrix)
    port_sharpe = calc_sharpe(best_w, mean_ret, cov_matrix)

    lines = []
    lines.append("Optimized SA - Single Objective Portfolio Optimisation Report")
    lines.append("=" * 64)
    lines.append(f"Date range    : {date_start} to {date_end}")
    lines.append(f"Trading days  : {trading_days}")
    lines.append(f"Seed          : {SEED}")
    lines.append(f"Risk-free rate: {RISK_FREE_RATE * 100:.2f}%")
    lines.append(f"Runs: {SA_RUNS}, Iterations: {SA_MAX_ITER}, Inner Loop (L): {SA_L}")
    lines.append("Objective: Maximize Sharpe Ratio")
    lines.append("")
    lines.append("Stock Universe:")
    for name in TICKERS.values():
        lines.append(f"  - {name}")
    lines.append(f"Max single-stock weight: {int(MAX_WEIGHT * 100)}%")
    lines.append("")

    # MODIFIED: Removed Max Drawdown from portfolio summary table
    port_table = [[
        "Optimized Portfolio",
        f"{port_return * 100:.2f}%",
        f"{port_risk * 100:.2f}%",
        round(port_sharpe, 4),
    ]]
    lines.append(tabulate(port_table,
                           headers=["Portfolio", "Return", "Risk", "Sharpe"],
                           tablefmt="pipe"))
    lines.append("")

    dim = len(ticker_list)
    stock_rows = []
    for i, ticker in enumerate(ticker_list):
        alloc = best_w[i]
        unit_w = np.zeros(dim)
        unit_w[i] = 1.0
        s_ret = calc_annual_return(unit_w, mean_ret)
        s_risk = float(np.sqrt(cov_matrix[i, i] * 252))
        s_sharpe = (s_ret - RISK_FREE_RATE) / max(s_risk, 0.05)
        alloc_str = f"{alloc * 100:.2f}%" if alloc >= 0.0001 else "0.00%"

        stock_rows.append([ticker, TICKERS[ticker], alloc_str,
                            f"{s_ret * 100:.2f}%", f"{s_risk * 100:.2f}%",
                            round(s_sharpe, 4), alloc])

    stock_rows.sort(key=lambda x: x[6], reverse=True)

    if runs_table is not None:
        lines.append(f"Per-Run Results (all {SA_RUNS} independent SA runs):")
        lines.append(runs_table)
        lines.append("")

    if summary_table is not None:
        lines.append("Run Statistics Summary (Mean / Min / Max / Std Dev):")
        lines.append(summary_table)
        lines.append("")

    # Removed Stock Return, Stock Risk, Stock Sharpe columns
    lines.append(tabulate([[r[0], r[1], r[2]] for r in stock_rows],
                           headers=["Ticker", "Stock Name", "Allocation"],
                           tablefmt="pipe"))
    lines.append("")

    return "\n".join(lines), stock_rows


def save_result(report_text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sa_portfolio_report_{timestamp}.txt"
    fpath = os.path.join(RESULTS_DIR, fname)

    with open(fpath, "w") as f:
        f.write(report_text)

    print(f"STEP 3: Report saved to results/{fname}\n")
    return fpath


def save_dashboard(best_w, close_df, returns_df, mean_ret, cov_matrix,
                    stock_rows, all_sharpes, best_trace, mean_trace, best_run_index):
    """
    Single combined chart, all three panels together:
    1) SA convergence curve (best run)
    2) Optimal portfolio weights bar chart
    3) Final Sharpe per run (best run highlighted, runs numbered 1-30)
    """
    ticker_list = list(TICKERS.keys())
    date_start = START_DATE
    date_end = END_DATE

    port_return = calc_annual_return(best_w, mean_ret)
    port_risk = calc_annual_risk(best_w, cov_matrix)
    port_sharpe = calc_sharpe(best_w, mean_ret, cov_matrix)
    final_w = repair_weights(best_w)

    base_colors = plt.cm.tab10.colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(ticker_list))]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5.5))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Optimized SA - Single Objective Portfolio Optimisation (Maximise Sharpe Ratio)\n"
        f"Data: S&P 500 Yahoo Finance {date_start} to {date_end}  |  "
        f"Seed: {SEED}  |  RF = {RISK_FREE_RATE*100:.1f}%",
        fontsize=14, fontweight="bold", y=1.04,
    )

    # Panel 1: convergence (best run)
    x_iter = np.arange(1, len(best_trace) + 1)
    ax1.plot(x_iter, best_trace, color="green", linewidth=2, label="Best Run Optimization")
    ax1.set_title(f"SA Convergence\n({SA_RUNS} runs, {SA_MAX_ITER} iterations)",
                   fontsize=13, fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: optimal weights bar chart
    allocs_pct = final_w * 100
    bars = ax2.bar(ticker_list, allocs_pct, color=colors, edgecolor="black", linewidth=0.6)
    for bar, alloc in zip(bars, allocs_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, alloc + 0.5,
                  f"{alloc:.1f}%", ha="center", va="bottom",
                  fontsize=9, fontweight="bold")
    ax2.set_title(
        "Optimal Portfolio Weights\n"
        f"Return={port_return*100:.2f}% | Risk={port_risk*100:.2f}% | "
        f"Sharpe={port_sharpe:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax2.set_ylabel("Allocation (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, max(allocs_pct.max() * 1.2, 5))

    # Panel 3: final Sharpe per run (runs numbered 1-30)
    sharpes = np.array(all_sharpes, dtype=float)
    run_numbers = np.arange(1, len(sharpes) + 1)
    mean_sh = sharpes.mean()
    bar_colors = ["orange" if i == best_run_index else "tab:blue" for i in range(len(sharpes))]

    ax3.bar(run_numbers, sharpes, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax3.axhline(mean_sh, color="mediumseagreen", linestyle="--", linewidth=1.5)

    pad = max((sharpes.max() - sharpes.min()) * 0.6, 1e-4)
    ax3.set_ylim(sharpes.min() - pad, sharpes.max() + pad)

    ax3.set_title("Final Sharpe per Run", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Run")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_xticks(run_numbers)
    ax3.set_xticklabels(run_numbers)
    ax3.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="orange", edgecolor="black", label=f"Best Run #{best_run_index + 1}"),
        Patch(facecolor="tab:blue", edgecolor="black", label="Other Runs"),
    ]
    ax3.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(GRAPHS_DIR, f"optimized_sa_result_{ts}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"STEP 4: Combined dashboard graph saved to graphs/optimized_sa_result_{ts}.png\n")
    return path


def main():
    print("SA Portfolio Optimizer | Yahoo Finance | 10 Stocks\n")

    close_df, returns_df = download_and_save()

    ticker_list = list(TICKERS.keys())
    close_df = close_df[ticker_list]
    returns_df = returns_df[ticker_list]
    returns_data = returns_df.values
    mean_ret = returns_data.mean(axis=0)
    cov_matrix = np.cov(returns_data.T)
    dim = len(ticker_list)

    best_w, all_sharpes, all_weights, best_trace, mean_trace, best_run_index = run_optimizer(mean_ret, cov_matrix, dim)

    runs_table, summary_table, run_returns, run_risks, run_sharpes = build_runs_report(
        all_sharpes, all_weights, close_df, mean_ret, cov_matrix
    )

    print(f"Per-Run Results (all {SA_RUNS} independent SA runs):")
    print(runs_table)
    print()
    print("Run Statistics Summary (Mean / Min / Max / Std Dev):")
    print(summary_table)
    print()

    report_text, stock_rows = build_report(
        best_w, close_df, returns_df, mean_ret, cov_matrix,
        runs_table=runs_table, summary_table=summary_table
    )
    print(report_text)

    save_result(report_text)

    save_dashboard(best_w, close_df, returns_df, mean_ret, cov_matrix,
                    stock_rows, all_sharpes, best_trace, mean_trace, best_run_index)

    print("Done!")
    print(f"data/    -> {len(os.listdir(DATA_DIR))} files")
    print(f"results/ -> {len(os.listdir(RESULTS_DIR))} file(s)")
    print(f"graphs/  -> {len(os.listdir(GRAPHS_DIR))} file(s)")


if __name__ == "__main__":
    main()

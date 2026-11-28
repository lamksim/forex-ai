"""The Phase 1 research loop: walk-forward evaluation with every trial logged,
validation gates, and a single sealed holdout evaluation.

    research(bars) -> for each fold: evaluate every config on the train window (logged),
                      pick the best by train Sharpe, run it on the next test window (logged)
                   -> stitch the out-of-sample windows into one walk-forward record
                   -> deflate its Sharpe by the number of configs ever tried
                   -> apply the gates in specs/performance_metrics.md
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import metrics as M
from .backtest import CostModel, run_backtest
from .registry import ExperimentLog, config_id
from .splits import HoldoutVault, walk_forward_folds
from .strategies import FAMILIES, generate


@dataclass(frozen=True)
class Gates:
    """Minimum evidence before a strategy may see the holdout (specs/performance_metrics.md)."""
    min_oos_sharpe: float = 1.0
    min_dsr: float = 0.95
    min_trades: int = 100
    min_profit_factor: float = 1.2
    max_drawdown: float = 0.05  # absolute value

    def check(self, m: dict, dsr: float) -> dict:
        pf = m.get("profit_factor") or 0.0
        return {
            f"oos_sharpe >= {self.min_oos_sharpe}": m["sharpe"] >= self.min_oos_sharpe,
            f"deflated_sharpe >= {self.min_dsr}": dsr >= self.min_dsr,
            f"trades >= {self.min_trades}": m["trades"] >= self.min_trades,
            f"profit_factor >= {self.min_profit_factor}": pf >= self.min_profit_factor,
            f"max_drawdown <= {self.max_drawdown:.0%}": abs(m["max_drawdown"]) <= self.max_drawdown,
        }


@dataclass
class ResearchResult:
    run_id: str
    symbol: str
    family: str
    folds: list[dict]
    oos_metrics: dict
    deflated: dict
    gates: dict
    passed: bool
    chosen_params: dict | None
    report_path: Path | None = None
    oos_equity: pd.Series = field(default=None, repr=False)


def _segment(bars: pd.DataFrame, start, end) -> pd.DataFrame:
    return bars[(bars.index >= start) & (bars.index < end)]


def evaluate(bars: pd.DataFrame, signal, stop, symbol, start, end, costs, equity=10_000.0):
    """Backtest one window. Signals are computed on the full history so indicators are
    warmed up; only bars inside the window are traded."""
    seg = _segment(bars, start, end)
    res = run_backtest(seg, signal, stop, symbol, initial_equity=equity, costs=costs)
    return res, M.summarize(res.equity, res.trades, equity)


def research(bars: pd.DataFrame, symbol: str, family: str, data_fp: str,
             log: ExperimentLog, costs: CostModel | None = None,
             train_months: int = 24, test_months: int = 6, gates: Gates | None = None,
             report_dir: Path | None = None) -> ResearchResult:
    costs = costs or CostModel()
    gates = gates or Gates()
    fam = FAMILIES[family]
    run_id = uuid.uuid4().hex[:8]
    folds = walk_forward_folds(bars.index, train_months, test_months)
    if not folds:
        raise ValueError(f"Not enough research data for a {train_months}+{test_months} month fold")

    cache = {config_id(family, p): (p, generate(family, bars, p, symbol)) for p in fam.configs()}
    fold_rows, oos_returns, oos_trades, chosen = [], [], [], None

    for k, f in enumerate(folds):
        best = None
        for cid, (params, (sig, stp)) in cache.items():
            _, m = evaluate(bars, sig, stp, symbol, f.train_start, f.train_end, costs)
            log.record(run_id=run_id, kind="train", symbol=symbol, family=family, params=params,
                       data_fingerprint=data_fp,
                       segment={"fold": k, "start": str(f.train_start), "end": str(f.train_end)},
                       metrics=m)
            if best is None or m["sharpe"] > best[2]["sharpe"]:
                best = (cid, params, m)
        cid, params, train_m = best
        sig, stp = cache[cid][1]
        res, test_m = evaluate(bars, sig, stp, symbol, f.train_end, f.test_end, costs)
        log.record(run_id=run_id, kind="test", symbol=symbol, family=family, params=params,
                   data_fingerprint=data_fp,
                   segment={"fold": k, "start": str(f.train_end), "end": str(f.test_end)},
                   metrics=test_m)
        oos_returns.append(M.daily_returns(res.equity))
        if len(res.trades):
            oos_trades.append(res.trades)
        fold_rows.append({"fold": k, "train": f"{f.train_start:%Y-%m-%d}..{f.train_end:%Y-%m-%d}",
                          "test": f"{f.train_end:%Y-%m-%d}..{f.test_end:%Y-%m-%d}",
                          "params": params, "train_sharpe": train_m["sharpe"],
                          "test_sharpe": test_m["sharpe"], "test_return": test_m["total_return"],
                          "test_trades": test_m["trades"], "halted": res.halted})
        chosen = params

    # Stitch the out-of-sample windows into one compounded equity curve.
    r = pd.concat(oos_returns).sort_index()
    oos_equity = 10_000.0 * (1 + r).cumprod()
    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame(
        columns=["pnl_usd", "commission_usd", "spread_cost_usd", "slippage_usd", "swap_usd",
                 "gross_mid_pnl_usd", "session", "exit_reason"])
    oos_m = M.summarize(oos_equity, trades, 10_000.0)
    oos_m["sharpe"] = M.sharpe(r)

    trials = log.trials(symbol)  # every config ever tried on this symbol, any family
    deflated = M.deflated_sharpe(r, max(len(trials), 1), list(trials.values()))
    g = gates.check(oos_m, deflated["dsr"])
    passed = all(g.values())

    # The config the lab would carry forward is the one the most recent fold chose.
    log.record(run_id=run_id, kind="walk_forward", symbol=symbol, family=family,
               params=chosen or {}, data_fingerprint=data_fp,
               segment={"folds": len(folds), "start": str(folds[0].train_end), "end": str(folds[-1].test_end)},
               metrics=oos_m, extra={"deflated": deflated, "gates": g, "passed": passed})

    result = ResearchResult(run_id, symbol, family, fold_rows, oos_m, deflated, g, passed,
                            chosen, oos_equity=oos_equity)
    if report_dir:
        result.report_path = write_report(result, fam.hypothesis, costs, Path(report_dir))
    return result


@dataclass
class PortfolioResult:
    family: str
    symbols: list[str]
    per_symbol: dict[str, dict]
    metrics: dict
    deflated: dict
    gates: dict
    passed: bool
    correlation: dict


def research_portfolio(results: list[ResearchResult], log: ExperimentLog, gates: Gates | None = None) -> PortfolioResult:
    """Combine per-pair walk-forward results of one family into a portfolio.

    Each pair risks 0.25% of the shared account per trade with at most one position,
    so the portfolio holds at most 3 positions (specs/risk_management.md). Its daily
    return is the sum of the pairs' out-of-sample daily returns. The trial count is
    every configuration tried on any of the pairs, so diversification cannot hide snooping.
    """
    gates = gates or Gates()
    fam = results[0].family
    rets = {r.symbol: r.oos_equity.pct_change().fillna(r.oos_equity.iloc[0] / 10_000.0 - 1) for r in results}
    frame = pd.DataFrame(rets).fillna(0.0)
    port = frame.sum(axis=1)
    equity = 10_000.0 * (1 + port).cumprod()
    m = {
        "sharpe": M.sharpe(port), "total_return": float(equity.iloc[-1] / 10_000.0 - 1),
        "max_drawdown": M.max_drawdown(equity), "avg_daily_return": float(port.mean()),
        "days": int(len(port)), "trades": int(sum(r.oos_metrics["trades"] for r in results)),
    }
    gross_win = sum(max(r.oos_metrics.get("net_profit_usd", 0), 0) for r in results)
    gross_loss = -sum(min(r.oos_metrics.get("net_profit_usd", 0), 0) for r in results)
    m["profit_factor"] = None  # per-trade profit factor is reported per pair
    trials: dict[str, float] = {}
    for r in results:
        trials.update(log.trials(r.symbol))
    deflated = M.deflated_sharpe(port, max(len(trials), 1), list(trials.values()))
    g = {k: v for k, v in gates.check({**m, "profit_factor": float("inf")}, deflated["dsr"]).items()
         if not k.startswith("profit_factor")}
    corr = frame.corr().round(2).to_dict()
    log.record(run_id="portfolio-" + "-".join(r.run_id for r in results), kind="portfolio",
               symbol="+".join(r.symbol for r in results), family=fam, params={},
               data_fingerprint="multi",
               segment={"start": str(frame.index[0]), "end": str(frame.index[-1])},
               metrics=m, extra={"deflated": deflated, "gates": g, "passed": all(g.values()),
                                 "pairs_net_usd": {r.symbol: r.oos_metrics.get("net_profit_usd") for r in results},
                                 "gross_win_usd": gross_win, "gross_loss_usd": gross_loss,
                                 "correlation": frame.corr().round(3).to_dict()})
    return PortfolioResult(fam, [r.symbol for r in results],
                           {r.symbol: {"sharpe": r.oos_metrics["sharpe"], "return": r.oos_metrics["total_return"],
                                       "dsr": r.deflated["dsr"]} for r in results},
                           m, deflated, g, all(g.values()), corr)


def write_portfolio_report(pr: PortfolioResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"PORTFOLIO_{pr.family}_{'-'.join(pr.symbols)}.md"
    m = pr.metrics
    lines = [f"# Portfolio report: {pr.family} on {', '.join(pr.symbols)}", "",
             f"**Verdict:** {'PASSED research gates' if pr.passed else 'REJECTED'}", "",
             "| Metric | Value |", "| --- | --- |"]
    for k in ("sharpe", "total_return", "max_drawdown", "avg_daily_return", "days", "trades"):
        lines.append(f"| {k} | {_fmt(m[k])} |")
    lines += ["", f"- Trials counted (all pairs): **{pr.deflated['n_trials']}**",
              f"- Best-of-N Sharpe from noise: **{pr.deflated['sr0_annual']:.2f}**",
              f"- Deflated Sharpe: **{pr.deflated['dsr']:.3f}**", "", "## Gates", ""]
    lines += [f"- [{'x' if ok else ' '}] {k}" for k, ok in pr.gates.items()]
    lines += ["", "## Pairs", "", "| Pair | Sharpe | Return | DSR |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {v['sharpe']:.2f} | {v['return']:.2%} | {v['dsr']:.3f} |" for k, v in pr.per_symbol.items()]
    lines += ["", "## Daily-return correlation", "", "| | " + " | ".join(pr.symbols) + " |",
              "| --- " * (len(pr.symbols) + 1) + "|"]
    for a in pr.symbols:
        lines.append(f"| {a} | " + " | ".join(f"{pr.correlation[b][a]:.2f}" for b in pr.symbols) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


def holdout_evaluation(all_bars: pd.DataFrame, holdout_start: pd.Timestamp, symbol: str,
                       family: str, params: dict, version: str, data_fp: str,
                       log: ExperimentLog, vault: HoldoutVault,
                       costs: CostModel | None = None) -> dict:
    """The one look at sealed data. Refused if this version or config already looked."""
    costs = costs or CostModel()
    cid = config_id(family, params)
    vault.open(symbol=symbol, version=version, config_id=cid, data_fingerprint=data_fp,
               reason="final validation after passing research gates")
    sig, stp = generate(family, all_bars, params, symbol)
    res, m = evaluate(all_bars, sig, stp, symbol, holdout_start, all_bars.index[-1] + pd.Timedelta(seconds=1), costs)
    log.record(run_id=f"holdout-{version}", kind="holdout", symbol=symbol, family=family,
               params=params, data_fingerprint=data_fp,
               segment={"start": str(holdout_start), "end": str(all_bars.index[-1])}, metrics=m)
    return m


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_report(r: ResearchResult, hypothesis: str, costs: CostModel, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{r.symbol}_{r.family}_{r.run_id}.md"
    m = r.oos_metrics
    lines = [
        f"# Walk-forward report: {r.symbol} / {r.family} (run {r.run_id})",
        "",
        f"**Hypothesis:** {hypothesis}",
        "",
        f"**Verdict:** {'PASSED research gates, eligible for one holdout evaluation' if r.passed else 'REJECTED: did not pass research gates'}",
        "",
        "## Out-of-sample results (stitched test windows, after costs)",
        "",
        "| Metric | Value |", "| --- | --- |",
    ]
    for k in ("total_return", "net_profit_usd", "sharpe", "max_drawdown", "avg_daily_return", "trades",
              "win_rate", "profit_factor", "avg_win_usd", "avg_loss_usd", "max_consecutive_losses",
              "gross_before_costs_usd", "total_costs_usd", "swap_usd", "net_trades_usd"):
        if k in m:
            lines.append(f"| {k} | {_fmt(m[k])} |")
    lines += [
        "",
        "## Data-snooping adjustment",
        "",
        f"- Configurations ever tried on {r.symbol} (from the experiment log): **{r.deflated['n_trials']}**",
        f"- Sharpe expected from the best of that many worthless strategies: **{r.deflated['sr0_annual']:.2f}** (annualized)",
        f"- Deflated Sharpe ratio (probability the edge is real): **{r.deflated['dsr']:.3f}**",
        "",
        "## Gates", "",
    ]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in r.gates.items()]
    lines += ["", "## Folds", "", "| Fold | Train | Test | Chosen params | Train Sharpe | Test Sharpe | Test return | Trades | Halted |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in r.folds:
        lines.append(f"| {f['fold']} | {f['train']} | {f['test']} | `{json.dumps(f['params'])}` | "
                     f"{f['train_sharpe']:.2f} | {f['test_sharpe']:.2f} | {f['test_return']:.2%} | "
                     f"{f['test_trades']} | {f['halted']} |")
    if m.get("by_session"):
        lines += ["", "## By session (out of sample)", "", "| Session | Trades | P&L USD |", "| --- | --- | --- |"]
        for s, v in m["by_session"].items():
            lines.append(f"| {s} | {v['trades']} | {v['pnl_usd']:.2f} |")
    lines += ["", "## Cost model", "", f"`{json.dumps(costs.describe())}` plus the bid/ask spread from the data.", ""]
    path.write_text("\n".join(lines))
    return path

"""Command line entry point: python -m forexlab <command>."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from . import data as D
from . import instruments
from .backtest import CostModel
from .registry import ExperimentLog
from .research import holdout_evaluation, research
from .splits import HoldoutVault, split_holdout
from .strategies import FAMILIES


def _load(args) -> tuple:
    bars = D.load_bars(args.data)
    if args.timeframe:
        bars = D.resample(bars, args.timeframe)
    fp = D.fingerprint(bars)
    research_bars, holdout_start = split_holdout(bars, args.holdout_frac)
    return bars, research_bars, holdout_start, fp


def _costs(args) -> CostModel:
    return CostModel(commission_per_100k_side=args.commission, slippage_pips=args.slippage,
                     min_spread_pips=args.min_spread, swap_long_pips=args.swap_long,
                     swap_short_pips=args.swap_short)


def cmd_portfolio(args):
    from .research import research_portfolio, write_portfolio_report
    log = ExperimentLog(Path(args.log))
    if len(args.data) != len(args.symbols):
        raise SystemExit("Give one --data file per --symbols entry, in the same order.")
    swaps = dict(x.split("=", 1) for x in args.swaps) if args.swaps else {}
    for fam in args.family or list(FAMILIES):
        results = []
        for sym, path in zip(args.symbols, args.data):
            bars = D.load_bars(path)
            if args.timeframe:
                bars = D.resample(bars, args.timeframe)
            fp = D.fingerprint(bars)
            rb, hs = split_holdout(bars, args.holdout_frac)
            sl, ss = (float(x) for x in swaps[sym].split(":")) if sym in swaps else (0.0, 0.0)
            costs = CostModel(args.commission, args.slippage, args.min_spread, sl, ss)
            results.append(research(rb, sym, fam, fp, log, costs, args.train_months, args.test_months,
                                    report_dir=Path(args.reports)))
        pr = research_portfolio(results, log)
        report = write_portfolio_report(pr, Path(args.reports))
        m = pr.metrics
        pairs = "  ".join(f"{k} {v['sharpe']:+.2f}" for k, v in pr.per_symbol.items())
        print(f"{fam:24s} PORTFOLIO sharpe {m['sharpe']:+.2f}  return {m['total_return']:+.2%}  "
              f"maxDD {m['max_drawdown']:.2%}  trades {m['trades']:4d}  DSR {pr.deflated['dsr']:.3f} "
              f"(N={pr.deflated['n_trials']})  {'PASS' if pr.passed else 'reject'}  | {pairs}  -> {report}")


def cmd_download(args):
    fn = D.download_dukascopy_h1 if args.bars == "H1" else D.download_dukascopy
    df = fn(args.symbol, date.fromisoformat(args.start), date.fromisoformat(args.end), Path(args.cache))
    out = Path(args.out or f"data/{args.symbol}_{args.bars}_{args.start}_{args.end}.csv.gz")
    D.save_bars(df, out)
    print(f"Saved {len(df):,} {args.bars} bars to {out} (fingerprint {D.fingerprint(df)})")


def cmd_mt5(args):
    df = D.load_mt5(args.symbol, args.start, args.end, args.timeframe, args.mt5_symbol, args.utc_offset, args.mt5_path)
    out = Path(args.out or f"data/{args.symbol}_{args.timeframe}_MT5_{args.start}_{args.end}.csv.gz")
    D.save_bars(df, out)
    sp = (df["ask_close"] - df["bid_close"]) / instruments.get(args.symbol).pip
    print(f"Saved {len(df):,} {args.timeframe} bars {df.index[0]} .. {df.index[-1]} to {out}; "
          f"median spread {sp.median():.2f} pips, p95 {sp.quantile(0.95):.2f}")
    if df.attrs.get("swap"):
        print(f"Current broker swap for {args.symbol}: {df.attrs['swap']} "
              "(today's rates only; use as an approximation via --swap-long/--swap-short)")


def cmd_synth(args):
    df = D.synthetic_bars(args.symbol, args.start, args.end, seed=args.seed)
    out = Path(args.out or f"data/{args.symbol}_synthetic_H1.csv.gz")
    D.save_bars(df, out)
    print(f"Saved {len(df):,} synthetic H1 bars to {out}")


def cmd_research(args):
    bars, rb, hs, fp = _load(args)
    print(f"Data {fp}: {bars.index[0]} .. {bars.index[-1]} ({len(bars):,} bars). "
          f"Holdout sealed from {hs:%Y-%m-%d}.")
    log = ExperimentLog(Path(args.log))
    families = args.family or list(FAMILIES)
    for fam in families:
        r = research(rb, args.symbol, fam, fp, log, _costs(args), args.train_months,
                     args.test_months, report_dir=Path(args.reports))
        m = r.oos_metrics
        print(f"{fam:14s} OOS sharpe {m['sharpe']:+.2f}  return {m['total_return']:+.2%}  "
              f"maxDD {m['max_drawdown']:.2%}  trades {m['trades']:4d}  "
              f"DSR {r.deflated['dsr']:.3f} (N={r.deflated['n_trials']})  "
              f"{'PASS' if r.passed else 'reject'}  -> {r.report_path}")


def cmd_holdout(args):
    bars, rb, hs, fp = _load(args)
    log = ExperimentLog(Path(args.log))
    # Only a configuration whose latest walk-forward run passed the gates may look.
    runs = [e for e in log.entries() if e["kind"] == "walk_forward" and e["symbol"] == args.symbol
            and e["family"] == args.family and e["data"] == fp]
    if not runs:
        raise SystemExit("No walk-forward run for this symbol/family/dataset. Run `research` first.")
    last = runs[-1]
    if not last.get("extra", {}).get("passed") and not args.force_failed:
        raise SystemExit("The latest walk-forward run did not pass the research gates. "
                         "The holdout is reserved for strategies that did (constitution rule 2).")
    # Freeze the version before the look, so it cannot be quietly changed afterwards.
    frozen = Path(args.versions) / f"{args.version}.json"
    if frozen.exists():
        raise SystemExit(f"{frozen} already exists. A new look needs a new version name.")
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({"version": args.version, "symbol": args.symbol, "family": args.family,
                                  "params": last["params"], "data": fp, "walk_forward_run": last["run_id"],
                                  "walk_forward_metrics": last["metrics"]}, indent=2, default=str))
    m = holdout_evaluation(bars, hs, args.symbol, args.family, last["params"], args.version, fp, log,
                           HoldoutVault(Path(args.vault)), _costs(args))
    m["passed_holdout"] = m["sharpe"] > 0.5 and abs(m["max_drawdown"]) <= 0.05
    print(json.dumps(m, indent=2, default=str))


def cmd_trials(args):
    log = ExperimentLog(Path(args.log))
    t = log.trials(args.symbol)
    print(f"{len(t)} distinct configurations tried on {args.symbol}; "
          f"{len(log.entries())} experiment records in {args.log}")


def cmd_live(args):
    import json as _json
    from . import live as L
    cfg = L.LiveConfig(symbol=args.symbol, broker_symbol=args.mt5_symbol, family=args.family,
                       params=_json.loads(args.params) if args.params else L.LiveConfig().params,
                       magic=args.magic, log_dir=Path(args.log_dir), utc_offset_hours=args.utc_offset)
    if args.once:
        import MetaTrader5 as mt5  # type: ignore
        ok = mt5.initialize(path=args.mt5_path, timeout=120_000) if args.mt5_path else mt5.initialize(timeout=120_000)
        if not ok:
            raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
        try:
            print(_json.dumps(L.run_once(mt5, cfg), default=str, indent=2))
        finally:
            mt5.shutdown()
    else:
        print(f"Forward-testing {cfg.family} {cfg.params} on {cfg.name} (DEMO only). "
              f"Logs: {cfg.log_dir}. Create {Path(cfg.log_dir) / 'STOP'} to flatten and stop.", flush=True)
        L.run_forever(cfg, terminal_path=args.mt5_path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="forexlab", description="Forex AI research lab, Phase 1")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download Dukascopy M1 bid/ask candles")
    d.add_argument("--symbol", default="EURUSD")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--cache", default="data/raw")
    d.add_argument("--bars", choices=["M1", "H1"], default="M1",
                   help="H1 needs ~25x fewer requests (one file per month); M1 sees spreads minute by minute")
    d.add_argument("--out")
    d.set_defaults(fn=cmd_download)

    m = sub.add_parser("mt5", help="Export bars from a running, logged-in MetaTrader 5 terminal (Windows)")
    m.add_argument("--symbol", default="EURUSD")
    m.add_argument("--mt5-symbol", help="Broker's name for the pair if it has a suffix, e.g. EURUSD.m")
    m.add_argument("--start", required=True)
    m.add_argument("--end", required=True)
    m.add_argument("--timeframe", default="H1", choices=["M1", "M5", "M15", "H1"])
    m.add_argument("--mt5-path", help=r"Path to terminal64.exe, e.g. C:\Program Files\MetaTrader 5\terminal64.exe")
    m.add_argument("--utc-offset", type=float, help="Fixed server-minus-UTC offset in hours. Default: New York time + 7 h (UTC+2/+3 with US daylight saving), which MetaQuotes-Demo and most FX brokers use")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_mt5)

    s = sub.add_parser("synth", help="Write a no-edge synthetic dataset for smoke tests")
    s.add_argument("--symbol", default="EURUSD")
    s.add_argument("--start", default="2019-01-01")
    s.add_argument("--end", default="2025-01-01")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_synth)

    for name, fn, help_ in (("research", cmd_research, "Walk-forward research, every trial logged"),
                            ("holdout", cmd_holdout, "One-time evaluation on the sealed holdout")):
        r = sub.add_parser(name, help=help_)
        r.add_argument("--data", required=True, help="Bar file from `download` or `synth`")
        r.add_argument("--symbol", default="EURUSD")
        r.add_argument("--timeframe", default="1h", help="Resample to this (empty string to keep as-is)")
        r.add_argument("--holdout-frac", type=float, default=0.2)
        r.add_argument("--commission", type=float, default=3.5, help="USD per 100k per side")
        r.add_argument("--slippage", type=float, default=0.2, help="pips per fill")
        r.add_argument("--min-spread", type=float, default=0.0, help="spread floor in pips")
        r.add_argument("--swap-long", type=float, default=0.0, help="pips per night for longs (negative = cost)")
        r.add_argument("--swap-short", type=float, default=0.0, help="pips per night for shorts (negative = cost)")
        r.add_argument("--log", default="research_log/experiments.jsonl")
        r.set_defaults(fn=fn)
    rp = sub.choices["research"]
    rp.add_argument("--family", action="append", choices=list(FAMILIES))
    rp.add_argument("--train-months", type=int, default=24)
    rp.add_argument("--test-months", type=int, default=6)
    rp.add_argument("--reports", default="research_log/reports")
    hp = sub.choices["holdout"]
    hp.add_argument("--family", required=True, choices=list(FAMILIES))
    hp.add_argument("--version", required=True, help="Strategy version label, e.g. ma_cross-v1")
    hp.add_argument("--vault", default="research_log/holdout_access.jsonl")
    hp.add_argument("--versions", default="strategy_versions")
    hp.add_argument("--force-failed", action="store_true",
                    help="Look anyway (logged; burns the holdout for this config)")

    pf = sub.add_parser("portfolio", help="Walk-forward each pair, then judge the 3-pair portfolio")
    pf.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"])
    pf.add_argument("--data", nargs="+", required=True, help="One bar file per symbol, same order")
    pf.add_argument("--swaps", nargs="*", help="Per pair long:short swap pips per night, e.g. EURUSD=-0.9:0.2 USDJPY=1.1:-2.0")
    pf.add_argument("--family", action="append", choices=list(FAMILIES))
    pf.add_argument("--timeframe", default="1h")
    pf.add_argument("--holdout-frac", type=float, default=0.2)
    pf.add_argument("--commission", type=float, default=3.5)
    pf.add_argument("--slippage", type=float, default=0.2)
    pf.add_argument("--min-spread", type=float, default=0.3)
    pf.add_argument("--train-months", type=int, default=24)
    pf.add_argument("--test-months", type=int, default=6)
    pf.add_argument("--reports", default="research_log/reports")
    pf.add_argument("--log", default="research_log/experiments.jsonl")
    pf.set_defaults(fn=cmd_portfolio)

    t = sub.add_parser("trials", help="How many configurations have been tried")
    t.add_argument("--symbol", default="EURUSD")
    t.add_argument("--log", default="research_log/experiments.jsonl")
    t.set_defaults(fn=cmd_trials)

    lv = sub.add_parser("live", help="Demo forward test on MT5: one decision per closed H1 bar (DEMO accounts only)")
    lv.add_argument("--symbol", default="USDJPY")
    lv.add_argument("--mt5-symbol", help="Broker's name for the pair if it has a suffix")
    lv.add_argument("--family", default="ma_cross_trending", choices=list(FAMILIES))
    lv.add_argument("--params", help='JSON, e.g. \'{"er_window": 120, "er_min": 0.15, "stop_atr": 3.0}\'')
    lv.add_argument("--magic", type=int, default=20261126, help="Only positions with this magic number are touched")
    lv.add_argument("--log-dir", default="trading_logs")
    lv.add_argument("--mt5-path")
    lv.add_argument("--utc-offset", type=float)
    lv.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    lv.set_defaults(fn=cmd_live)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

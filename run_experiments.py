"""Batch-Läufe: alle Regime mit denselben Seeds (Common Random Numbers).

Beispiele:
    python run_experiments.py                    # Standard: 30 Läufe je Regime
    python run_experiments.py --runs 100 --workers 8
    python run_experiments.py --sweep            # zusätzlich Sensitivitätsläufe
    python run_experiments.py --sweep-only --runs 10
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from config import REGIMES, make_params
from metrics import lorenz, summarize_run
from model import Economy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Sensitivitätsläufe: (Name, Regime, Parameter, Werte)
#  - Eingriffsstärke: Wie verändert sich das Ergebnis, wenn man den Regler dreht?
#  - Robustheit: Hängt die Rangfolge der Regime an einer Verhaltensannahme?
SWEEPS = [
    ("preisdeckel", ["C_eingriff"], "price_cap", [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]),
    ("mindestlohn", ["C_eingriff"], "min_wage_ratio", [0.6, 0.7, 0.8, 0.85, 0.9]),
    ("steuersatz", ["B_rahmen"], "tax_rate", [0.0, 0.1, 0.2, 0.3, 0.4]),
    ("marktmacht", list(REGIMES), "market_power", [0.0, 1.0, 2.0, 4.0]),
    ("betrug", list(REGIMES), "fraud_prob", [0.0, 0.15, 0.3, 0.5]),
    ("konsum_vermoegen", list(REGIMES), "c_wealth", [0.02, 0.05, 0.1]),
    ("markup_tempo", list(REGIMES), "markup_step", [0.005, 0.01, 0.02]),
    # Kreditmarkt
    ("kredit", list(REGIMES), "credit", [False, True]),
    ("zinsdeckel", ["C_eingriff"], "interest_cap", [0.005, 0.006, 0.008, 0.012, 0.1]),
    ("eigenkapital", list(REGIMES), "capital_requirement", [0.02, 0.04, 0.08, 0.16]),
]


def _run(job):
    regime, seed, overrides = job
    p = make_params(regime, **overrides)
    eco = Economy(p, seed)
    ts = eco.run()
    summary = summarize_run(ts, p)
    grid, lc = lorenz(eco.wealth())
    return regime, seed, overrides, ts, summary, lc


def _pool(jobs, workers):
    if workers <= 1:
        return [_run(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, jobs, chunksize=1))


def run_main(runs: int, workers: int):
    jobs = [(r, s, {}) for s in range(runs) for r in REGIMES]
    t0 = time.time()
    res = _pool(jobs, workers)
    ts_all, summ, lorenz_rows = [], [], []
    for regime, seed, _, ts, summary, lc in res:
        ts = ts.assign(regime=regime, seed=seed)
        ts_all.append(ts)
        summ.append({"regime": regime, "seed": seed, **summary})
        lorenz_rows.append({"regime": regime, "seed": seed, **{f"q{i}": v for i, v in enumerate(lc)}})
    os.makedirs(OUT, exist_ok=True)
    pd.concat(ts_all).to_csv(os.path.join(OUT, "zeitreihen.csv.gz"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "kennzahlen.csv"), index=False)
    pd.DataFrame(lorenz_rows).to_csv(os.path.join(OUT, "lorenz.csv"), index=False)
    print(f"Hauptexperiment: {len(jobs)} Läufe in {time.time() - t0:.0f} s -> {OUT}")


def run_sweeps(runs: int, workers: int):
    """Jeder Sweep wird einzeln gespeichert (results/sweeps/); fertige werden übersprungen.
    So lässt sich ein abgebrochener Durchlauf einfach fortsetzen."""
    part_dir = os.path.join(OUT, "sweeps")
    os.makedirs(part_dir, exist_ok=True)
    t0 = time.time()
    for name, regimes, param, values in SWEEPS:
        path = os.path.join(part_dir, f"{name}.csv")
        if os.path.exists(path):
            print(f"  {name}: vorhanden, übersprungen", flush=True)
            continue
        jobs = [(r, s, {param: v}) for v in values for r in regimes for s in range(runs)]
        rows = []
        for regime, seed, ov, _, summary, _ in _pool(jobs, workers):
            rows.append({"sweep": name, "parameter": param, "wert": ov[param],
                         "regime": regime, "seed": seed, **summary})
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print(f"  {name}: {len(jobs)} Läufe fertig ({time.time() - t0:.0f} s)", flush=True)
    parts = [pd.read_csv(os.path.join(part_dir, f"{n}.csv")) for n, *_ in SWEEPS]
    pd.concat(parts).to_csv(os.path.join(OUT, "sensitivitaet.csv"), index=False)
    print(f"Sensitivität komplett ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30, help="Läufe (Seeds) je Regime")
    ap.add_argument("--sweep-runs", type=int, default=None, help="Läufe je Sweep-Punkt (Standard: runs/3)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--sweep", action="store_true", help="zusätzlich Sensitivitätsläufe")
    ap.add_argument("--sweep-only", action="store_true", help="nur Sensitivitätsläufe")
    a = ap.parse_args()
    sweep_runs = a.sweep_runs or max(3, a.runs // 3)
    if not a.sweep_only:
        run_main(a.runs, a.workers)
    if a.sweep or a.sweep_only:
        run_sweeps(sweep_runs, a.workers)

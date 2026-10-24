"""Auswertung: Diagramme und Ergebnistabelle aus results/.

    python analysis.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REGIMES, REGIME_LABELS, Params
from metrics import KPIS

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")

# Farben (geprüft auf Farbfehlsichtigkeit) plus zweite Kodierung über Linienstil/Marker
COLORS = {"A_frei": "#e34948", "A_eigentum": "#eda100", "B_rahmen": "#2a78d6", "C_eingriff": "#1baf7a"}
STYLES = {"A_frei": "-", "A_eigentum": "--", "B_rahmen": "-", "C_eingriff": "-."}
MARKERS = {"A_frei": "o", "A_eigentum": "s", "B_rahmen": "D", "C_eingriff": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.frameon": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.dpi": 150, "savefig.bbox": "tight",
})

TS_PANELS = [
    ("konsum_pk", "Konsum pro Kopf", None),
    ("arbeitslosigkeit", "Arbeitslosenquote", "pct"),
    ("knappheit", "Unbefriedigte Nachfrage", "pct"),
    ("gini_vermoegen", "Gini Vermögen", None),
    ("preisindex", "Preisniveau", None),
    ("hhi", "Marktkonzentration (HHI)", None),
]


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def _pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))


def _legend_top(fig, regimes):
    handles = [plt.Line2D([], [], color=COLORS[r], ls=STYLES[r], marker=MARKERS[r], lw=2, ms=6)
               for r in regimes]
    fig.legend(handles, [REGIME_LABELS[r] for r in regimes], loc="upper center",
               ncol=len(regimes), bbox_to_anchor=(0.5, 1.02))


def fig_timeseries(ts, p):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True)
    ts = ts[ts.t >= 150]
    for ax, (col, title, fmt) in zip(axes.flat, TS_PANELS):
        ax.axvspan(p.shock_period, p.shock_period + p.shock_duration, color=GRID, alpha=0.7, lw=0)
        ax.axvline(p.burn_in, color=MUTED, lw=0.8, ls=":")
        for r in REGIMES:
            g = ts[ts.regime == r].groupby("t")[col]
            m, s, n = g.mean(), g.std(), g.count()
            half = 1.96 * s / np.sqrt(n)
            ax.fill_between(m.index, m - half, m + half, color=COLORS[r], alpha=0.18, lw=0)
            ax.plot(m.index, m.values, color=COLORS[r], ls=STYLES[r], lw=1.6)
        ax.set_title(title, loc="left")
        if fmt == "pct":
            _pct(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Periode")
    axes[0, 0].annotate("Schock", xy=(p.shock_period + p.shock_duration / 2, axes[0, 0].get_ylim()[0]),
                        xytext=(0, 4), textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    axes[0, 0].annotate("Ende Einschwingen", xy=(p.burn_in, axes[0, 0].get_ylim()[1]), xytext=(3, -10),
                        textcoords="offset points", color=MUTED, fontsize=8)
    _legend_top(fig, REGIMES)
    fig.text(0.5, -0.02, "Linie = Mittelwert über alle Läufe, Band = 95-%-Konfidenzintervall, "
             "grau = Produktivitätsschock", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_zeitreihen.png"))
    plt.close(fig)


def fig_shock(ts, p):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    win = ts[(ts.t >= p.shock_period - 20) & (ts.t < min(p.shock_period + 150, p.periods))]
    for r in REGIMES:
        d = ts[(ts.regime == r)]
        ref = d[(d.t >= p.burn_in) & (d.t < p.shock_period)].groupby("seed").konsum_pk.mean()
        w = win[win.regime == r].copy()
        w["rel"] = w.konsum_pk / w.seed.map(ref)
        g = w.groupby("t").rel
        m, half = g.mean(), g.apply(ci95)
        ax.fill_between(m.index, m - half, m + half, color=COLORS[r], alpha=0.18, lw=0)
        ax.plot(m.index, m.values, color=COLORS[r], ls=STYLES[r], lw=2, label=REGIME_LABELS[r])
    ax.axvspan(p.shock_period, p.shock_period + p.shock_duration, color=GRID, alpha=0.7, lw=0)
    ax.axhline(1, color=MUTED, lw=0.8)
    _pct(ax)
    ax.set_title("Konsum relativ zum Niveau vor dem Schock", loc="left")
    ax.set_xlabel("Periode")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(os.path.join(FIG, "02_schock.png"))
    plt.close(fig)


def fig_lorenz():
    lz = pd.read_csv(os.path.join(RES, "lorenz.csv"))
    qcols = [c for c in lz.columns if c.startswith("q")]
    grid = np.linspace(0, 1, len(qcols))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.text(0.55, 0.6, "Gleichverteilung", rotation=42, color=MUTED, fontsize=8)
    for r in REGIMES:
        m = lz[lz.regime == r][qcols].mean().to_numpy()
        ax.plot(grid, m, color=COLORS[r], ls=STYLES[r], lw=2, label=REGIME_LABELS[r])
    ax.set_xlabel("Bevölkerungsanteil (aufsteigend nach Vermögen)")
    ax.set_ylabel("Anteil am Gesamtvermögen")
    _pct(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))
    ax.set_title("Vermögensverteilung am Ende (Lorenzkurven)", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(os.path.join(FIG, "03_lorenz.png"))
    plt.close(fig)


def fig_tradeoff(km):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in REGIMES:
        d = km[km.regime == r]
        ax.scatter(d.gini_vermoegen, d.wohlfahrt, s=22, marker=MARKERS[r], color=COLORS[r],
                   alpha=0.45, edgecolor="none")
        mx, my = d.gini_vermoegen.mean(), d.wohlfahrt.mean()
        ax.errorbar(mx, my, xerr=ci95(d.gini_vermoegen), yerr=ci95(d.wohlfahrt), fmt=MARKERS[r],
                    ms=10, color=COLORS[r], mec="#fcfcfb", mew=1.5, elinewidth=1.5, capsize=0)
        off = {"A_frei": (10, 6), "A_eigentum": (10, 12), "B_rahmen": (-10, -18), "C_eingriff": (12, -16)}[r]
        ax.annotate(REGIME_LABELS[r], (mx, my), xytext=off, textcoords="offset points",
                    fontsize=9, color=INK, ha="right" if off[0] < 0 else "left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
    ax.set_xlabel("Gini-Koeffizient Vermögen  (← gleicher)")
    ax.set_ylabel("Wohlfahrt, Ø log(1 + Konsum)  (höher →)")
    ax.set_title("Effizienz und Gleichheit: jeder Punkt ein Lauf, groß = Mittelwert", loc="left")
    fig.savefig(os.path.join(FIG, "04_effizienz_gleichheit.png"))
    plt.close(fig)


def fig_profile(km):
    keys = list(KPIS)
    fig, axes = plt.subplots(len(keys) // 2, 2, figsize=(11, 1.15 * len(keys) / 2 + 1.2))
    regs = list(REGIMES)
    for ax, k in zip(axes.T.flat, keys):
        label, higher = KPIS[k]
        for i, r in enumerate(regs):
            v = km[km.regime == r][k]
            ax.errorbar(v.mean(), i, xerr=ci95(v), fmt=MARKERS[r], color=COLORS[r], ms=7, elinewidth=1.5)
        ax.set_yticks(range(len(regs)))
        ax.set_yticklabels([r.split("_")[0] + (" frei" if r == "A_frei" else " Eig." if r == "A_eigentum" else "")
                            for r in regs], fontsize=8)
        ax.set_ylim(len(regs) - 0.4, -0.6)
        hint = "keine Wertung" if higher is None else ("höher" if higher else "niedriger") + " = günstiger"
        ax.set_title(f"{label}  ({hint})", loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    _legend_top(fig, regs)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_kennzahlenprofil.png"))
    plt.close(fig)


def fig_sweeps():
    path = os.path.join(RES, "sensitivitaet.csv")
    if not os.path.exists(path):
        return
    sw = pd.read_csv(path)
    # Eingriffsstärke: mehrere Kennzahlen gegen den Regler
    inten = [("preisdeckel", "Preisdeckel (max. Aufschlag auf Stückkosten)"),
             ("mindestlohn", "Mindestlohn (Anteil am Wertgrenzprodukt)"),
             ("steuersatz", "Steuersatz in B")]
    show = [("konsum_pk", "Konsum pro Kopf", False), ("knappheit", "Unbefr. Nachfrage", True),
            ("arbeitslosigkeit", "Arbeitslosigkeit", True), ("schockverlust", "Konsumverlust nach Schock", False)]
    fig, axes = plt.subplots(len(inten), len(show), figsize=(13, 2.6 * len(inten)))
    for row, (name, xlabel) in zip(axes, inten):
        d = sw[sw.sweep == name]
        if d.empty:
            continue
        reg = d.regime.iloc[0]
        for ax, (k, lab, pct) in zip(row, show):
            g = d.groupby("wert")[k]
            m, h = g.mean(), g.apply(ci95)
            ax.fill_between(m.index, m - h, m + h, color=COLORS[reg], alpha=0.18, lw=0)
            ax.plot(m.index, m.values, color=COLORS[reg], marker=MARKERS[reg], lw=2, ms=5)
            ax.set_title(lab, loc="left", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            if pct:
                _pct(ax)
    fig.suptitle("Eingriffsstärke: Wie reagieren die Kennzahlen auf den Regler?", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_eingriffsstaerke.png"))
    plt.close(fig)

    # Robustheit: Wohlfahrt aller Regime gegen Verhaltensannahmen
    rob = [("marktmacht", "Marktmacht ohne Kartellaufsicht"), ("betrug", "Betrugsneigung"),
           ("konsum_vermoegen", "Konsumneigung aus Vermögen"), ("markup_tempo", "Tempo der Preisanpassung")]
    fig, axes = plt.subplots(2, len(rob), figsize=(14, 6.4))
    for col, (name, xlabel) in enumerate(rob):
        d = sw[sw.sweep == name]
        for row, (k, lab) in enumerate([("wohlfahrt", "Wohlfahrt"), ("gini_vermoegen", "Gini Vermögen")]):
            ax = axes[row, col]
            for r in REGIMES:
                g = d[d.regime == r].groupby("wert")[k]
                if g.ngroups == 0:
                    continue
                m, h = g.mean(), g.apply(ci95)
                ax.fill_between(m.index, m - h, m + h, color=COLORS[r], alpha=0.15, lw=0)
                ax.plot(m.index, m.values, color=COLORS[r], ls=STYLES[r], marker=MARKERS[r], lw=1.8, ms=5)
            ax.set_title(f"{lab}", loc="left", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
    _legend_top(fig, REGIMES)
    fig.text(0.01, -0.03, "Robust ist eine Aussage nur, wenn die Rangfolge der Regime über die "
             "gesamte Achse erhalten bleibt.", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "07_robustheit.png"))
    plt.close(fig)


CREDIT_PANELS = [
    ("kreditquote", "Kredite / Jahresproduktion", "pct"),
    ("zins", "Ø Kreditzins je Periode", "pct"),
    ("rationierung", "Abgelehnte Kreditanträge", "pct"),
    ("ausfaelle", "Kreditausfälle je Periode", "pct"),
    ("bank_ek_quote", "Eigenkapital / Kredite der Bank", None),
    ("geldmenge", "Geldmenge (Buchgeld)", None),
]


def fig_credit_ts(ts, p):
    if "kredite" not in ts:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True)
    ts = ts[ts.t >= 150]
    for ax, (col, title, fmt) in zip(axes.flat, CREDIT_PANELS):
        ax.axvspan(p.shock_period, p.shock_period + p.shock_duration, color=GRID, alpha=0.7, lw=0)
        for r in REGIMES:
            d = ts[ts.regime == r]
            if col == "ausfaelle":  # verrauscht: gleitender Durchschnitt über 10 Perioden
                d = d.assign(ausfaelle=d.groupby("seed").ausfaelle.transform(
                    lambda x: x.rolling(10, min_periods=1).mean()))
            g = d.groupby("t")[col]
            m, h = g.mean(), g.apply(ci95)
            ax.fill_between(m.index, m - h, m + h, color=COLORS[r], alpha=0.18, lw=0)
            ax.plot(m.index, m.values, color=COLORS[r], ls=STYLES[r], lw=1.6)
        ax.set_title(title, loc="left")
        if fmt == "pct":
            _pct(ax)
        if col == "bank_ek_quote":
            ax.set_yscale("log")
            ax.axhline(0.08, color=MUTED, lw=0.8, ls=":")
            ax.annotate("8 % Mindestquote B/C", (ts.t.min(), 0.08), xytext=(2, 3),
                        textcoords="offset points", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("Periode")
    _legend_top(fig, REGIMES)
    fig.text(0.5, -0.02, "Mittelwert über alle Läufe mit 95-%-Band; Ausfälle als gleitender "
             "10-Perioden-Durchschnitt; grau = Produktivitätsschock", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "08_kreditmarkt.png"))
    plt.close(fig)


def fig_credit_compare():
    path = os.path.join(RES, "sensitivitaet.csv")
    if not os.path.exists(path):
        return
    sw = pd.read_csv(path)
    d = sw[sw.sweep == "kredit"].copy()
    if d.empty:
        return
    d["wert"] = d.wert.astype(str).str.lower().map({"true": True, "false": False, "1.0": True, "0.0": False})
    keys = [("konsum_pk", "Konsum pro Kopf"), ("arbeitslosigkeit", "Arbeitslosenquote"),
            ("gini_vermoegen", "Gini Vermögen"), ("schockverlust", "Konsumverlust nach Schock"),
            ("insolvenzen", "Insolvenzen je 100 Perioden"), ("volatilitaet", "Volatilität Konsum")]
    regs = list(REGIMES)
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
    for ax, (k, lab) in zip(axes.flat, keys):
        for i, r in enumerate(regs):
            for credit, face in ((False, "none"), (True, COLORS[r])):
                v = d[(d.regime == r) & (d.wert == credit)][k]
                ax.errorbar(v.mean(), i + (0.15 if credit else -0.15), xerr=ci95(v), fmt=MARKERS[r],
                            color=COLORS[r], mfc=face, ms=7, elinewidth=1.4)
        ax.set_yticks(range(len(regs)))
        ax.set_yticklabels([REGIME_LABELS[r].replace(" (", "\n(") for r in regs], fontsize=7.5)
        ax.set_ylim(len(regs) - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(lab, loc="left", fontsize=10)
        if k == "arbeitslosigkeit":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))
    handles = [plt.Line2D([], [], color=MUTED, marker="o", mfc="none", ls="", ms=7),
               plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=7)]
    fig.legend(handles, ["ohne Kreditmarkt", "mit Kreditmarkt"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "09_mit_ohne_kredit.png"))
    plt.close(fig)

    # Regulierung: Zinsobergrenze (C) und Eigenkapitalanforderung (alle)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    z = sw[sw.sweep == "zinsdeckel"]
    zk = [("rationierung", "Abgelehnte Kreditanträge", True), ("kreditquote", "Kredite / Jahresproduktion", True),
          ("arbeitslosigkeit", "Arbeitslosenquote", True), ("schockverlust", "Konsumverlust nach Schock", False)]
    for ax, (k, lab, pct) in zip(axes[0], zk):
        if z.empty:
            break
        g = z.groupby("wert")[k]
        m, h = g.mean(), g.apply(ci95)
        x = np.arange(len(m))
        ax.fill_between(x, m - h, m + h, color=COLORS["C_eingriff"], alpha=0.18, lw=0)
        ax.plot(x, m.values, color=COLORS["C_eingriff"], marker=MARKERS["C_eingriff"], lw=2, ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels([("kein" if v >= 0.1 else f"{v * 100:.1f} %") for v in m.index], fontsize=8)
        ax.set_xlabel("Zinsobergrenze in C (je Periode)", fontsize=8)
        ax.set_title(lab, loc="left", fontsize=9)
        if pct:
            _pct(ax)
    e = sw[sw.sweep == "eigenkapital"]
    ek = [("bankenkrisen", "Bankenkrisen je 100 Perioden", False), ("rationierung", "Abgelehnte Kreditanträge", True),
          ("konsum_pk", "Konsum pro Kopf", False), ("schockverlust", "Konsumverlust nach Schock", False)]
    for ax, (k, lab, pct) in zip(axes[1], ek):
        for r in REGIMES:
            g = e[e.regime == r].groupby("wert")[k]
            if g.ngroups == 0:
                continue
            m, h = g.mean(), g.apply(ci95)
            ax.fill_between(m.index, m - h, m + h, color=COLORS[r], alpha=0.15, lw=0)
            ax.plot(m.index, m.values, color=COLORS[r], ls=STYLES[r], marker=MARKERS[r], lw=1.8, ms=5)
        ax.set_xlabel("Eigenkapitalanforderung der Bank", fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(lab, loc="left", fontsize=9)
        if pct:
            _pct(ax)
    _legend_top(fig, REGIMES)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "10_kreditregulierung.png"))
    plt.close(fig)


def credit_table(lines):
    path = os.path.join(RES, "sensitivitaet.csv")
    if not os.path.exists(path):
        return lines
    sw = pd.read_csv(path)
    d = sw[sw.sweep == "kredit"].copy()
    if d.empty:
        return lines
    d["wert"] = d.wert.astype(str).str.lower().map({"true": True, "false": False, "1.0": True, "0.0": False})
    lines += ["", "## Wirkung des Kreditmarkts (gepaart: mit − ohne Kredit, gleicher Seed)", "",
              f"Aus den Sensitivitätsläufen ({d.seed.nunique()} Seeds je Regime); * = 95-%-Intervall schließt 0 aus.", "",
              "| Kennzahl | " + " | ".join(REGIME_LABELS[r] for r in REGIMES) + " |",
              "|---" * (len(REGIMES) + 1) + "|"]
    for k, (label, _) in KPIS.items():
        if k in ("kreditquote", "zins", "rationierung", "ausfallquote", "bankenkrisen", "bankhilfen"):
            continue
        cells = []
        for r in REGIMES:
            on = d[(d.regime == r) & d.wert].set_index("seed")[k]
            off = d[(d.regime == r) & ~d.wert].set_index("seed")[k]
            diff = (on - off).dropna()
            h = ci95(diff)
            star = "*" if not np.isnan(h) and abs(diff.mean()) > h else ""
            cells.append(f"{diff.mean():+.3f}{star}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def table(km):
    lines = ["# Ergebnisse", "",
             "Mittelwert ± 95-%-Konfidenzintervall über alle Läufe (ruhige Phase zwischen "
             "Einschwingen und Schock; Schockkennzahlen danach).", ""]
    head = "| Kennzahl | " + " | ".join(REGIME_LABELS[r] for r in REGIMES) + " |"
    lines += [head, "|---" * (len(REGIMES) + 1) + "|"]
    for k, (label, higher) in KPIS.items():
        cells = []
        for r in REGIMES:
            v = km[km.regime == r][k]
            miss = v.isna().sum()
            cell = f"{v.mean():.3f} ± {ci95(v):.3f}"
            if miss:
                cell += f" ({miss} ohne Erholung)"
            cells.append(cell)
        arrow = "" if higher is None else (" ↑" if higher else " ↓")
        lines.append(f"| {label}{arrow} | " + " | ".join(cells) + " |")

    # Gepaarte Differenzen (gleicher Seed): nutzt die Common Random Numbers
    lines += ["", "## Gepaarte Differenzen zu B (gleicher Seed)", "",
              "Differenz = Regime − B, je Seed gebildet; * = 95-%-Intervall schließt 0 aus.", ""]
    others = [r for r in REGIMES if r != "B_rahmen"]
    lines += ["| Kennzahl | " + " | ".join(REGIME_LABELS[r] for r in others) + " |",
              "|---" * (len(others) + 1) + "|"]
    b = km[km.regime == "B_rahmen"].set_index("seed")
    for k, (label, _) in KPIS.items():
        cells = []
        for r in others:
            d = (km[km.regime == r].set_index("seed")[k] - b[k]).dropna()
            h = ci95(d)
            star = "*" if not np.isnan(h) and abs(d.mean()) > h else ""
            cells.append(f"{d.mean():+.3f}{star}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines = credit_table(lines)
    lines += ["", "Die Pfeile geben eine Darstellungskonvention an (↑ höher, ↓ niedriger = günstiger). "
              "Eine Gesamtrangfolge wird bewusst nicht gebildet: Die Gewichtung der Ziele ist eine "
              "normative Entscheidung und kein Modellergebnis."]
    with open(os.path.join(BASE, "ERGEBNISSE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    p = Params()
    ts = pd.read_csv(os.path.join(RES, "zeitreihen.csv.gz"))
    km = pd.read_csv(os.path.join(RES, "kennzahlen.csv"))
    fig_timeseries(ts, p)
    fig_shock(ts, p)
    fig_lorenz()
    fig_tradeoff(km)
    fig_profile(km)
    fig_sweeps()
    fig_credit_ts(ts, p)
    fig_credit_compare()
    table(km)
    print(f"Diagramme in {FIG}, Tabelle in ERGEBNISSE.md")


if __name__ == "__main__":
    main()

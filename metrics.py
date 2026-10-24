"""Kennzahlen: Ungleichheit, Konzentration und Zusammenfassung eines Laufs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Params


def gini(x) -> float:
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def hhi(shares) -> float:
    s = np.asarray(shares, dtype=float)
    tot = s.sum()
    return float(((s / tot) ** 2).sum()) if tot > 0 else 0.0


def lorenz(x, points: int = 101):
    """Lorenzkurve auf einem festen Raster (für Mittelung über Läufe)."""
    x = np.sort(np.clip(np.asarray(x, dtype=float), 0, None))
    cum = np.concatenate([[0], np.cumsum(x)])
    cum = cum / cum[-1] if cum[-1] > 0 else cum
    pop = np.linspace(0, 1, len(cum))
    grid = np.linspace(0, 1, points)
    return grid, np.interp(grid, pop, cum)


# Kennzahlen der Zusammenfassung: Spalte -> (Beschriftung, höher ist günstiger?)
# Die Richtung ist eine Darstellungskonvention, keine Wertung des Modells.
KPIS = {
    "konsum_pk": ("Konsum pro Kopf", True),
    "wohlfahrt": ("Wohlfahrt (Ø log-Nutzen)", True),
    "arbeitslosigkeit": ("Arbeitslosenquote", False),
    "knappheit": ("Unbefriedigte Nachfrage", False),
    "gini_vermoegen": ("Gini Vermögen", False),
    "gini_einkommen": ("Gini Einkommen", False),
    "hhi": ("Marktkonzentration (HHI)", False),
    "betrugsquote": ("Betrugsverluste (Anteil)", False),
    "volatilitaet": ("Volatilität Konsum (VK)", False),
    "inflation_vol": ("Volatilität Inflation", False),
    "schockverlust": ("Kumul. Konsumverlust nach Schock", False),
    "erholungszeit": ("Erholungszeit (Perioden)", False),
    "insolvenzen": ("Insolvenzen je 100 Perioden", False),
    "staatsquote": ("Anteil Staatsbeschäftigte", False),
    # Kreditmarkt (None = keine Richtung, beschreibende Größe)
    "kreditquote": ("Kredite / Jahresproduktion", None),
    "zins": ("Ø Kreditzins je Periode", None),
    "rationierung": ("Abgelehnte Kreditanträge", None),
    "ausfallquote": ("Kreditausfälle je Periode", False),
    "bankenkrisen": ("Bankenkrisen je 100 Perioden", False),
    "bankhilfen": ("Staatshilfe + Einlegerverluste (Anteil Geldmenge)", False),
}


def summarize_run(ts: pd.DataFrame, p: Params) -> dict:
    """Verdichtet die Zeitreihe eines Laufs zu Kennzahlen.

    Niveaus: ruhige Phase zwischen Einschwingen und Schock.
    Schock: Konsumverlust und Erholung relativ zum Niveau vor dem Schock.
    """
    t = ts["t"].to_numpy()
    steady = ts[(t >= p.burn_in) & (t < p.shock_period)]
    post = ts[t >= p.burn_in]  # Krisenzählung einschließlich Schockphase
    c = ts["konsum_pk"].to_numpy()
    ref = steady["konsum_pk"].mean()

    win = (t >= p.shock_period) & (t < min(p.shock_period + 100, p.periods))
    loss = float(np.clip(ref - c[win], 0, None).sum() / ref) if ref > 0 else np.nan

    end = p.shock_period + p.shock_duration
    roll = pd.Series(c).rolling(10, min_periods=10).mean().to_numpy()
    rec = np.nan
    for i in range(end, p.periods):
        if roll[i] >= 0.98 * ref:
            rec = float(i - end)
            break

    return {
        "konsum_pk": steady["konsum_pk"].mean(),
        "wohlfahrt": steady["wohlfahrt"].mean(),
        "arbeitslosigkeit": steady["arbeitslosigkeit"].mean(),
        "knappheit": steady["knappheit"].mean(),
        "gini_vermoegen": steady["gini_vermoegen"].mean(),
        "gini_einkommen": steady["gini_einkommen"].mean(),
        "hhi": steady["hhi"].mean(),
        "betrugsquote": steady["betrugsquote"].mean(),
        "volatilitaet": steady["konsum_pk"].std() / ref if ref > 0 else np.nan,
        "inflation_vol": steady["inflation"].std(),
        "schockverlust": loss,
        "erholungszeit": rec,
        "insolvenzen": steady["insolvenzen"].sum() / len(steady) * 100,
        "staatsquote": steady["staatsquote"].mean(),
        "kreditquote": steady["kreditquote"].mean(),
        "zins": steady["zins"].mean(),
        "rationierung": steady["rationierung"].mean(),
        "ausfallquote": steady["ausfaelle"].mean(),
        "bankenkrisen": post["bankenkrise"].sum() / len(post) * 100,
        "bankhilfen": float((post["bankenrettung"] + post["bail_in"]).sum() / post["geldmenge"].mean()),
    }

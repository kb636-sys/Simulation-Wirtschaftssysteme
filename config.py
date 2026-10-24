"""Parameter und Regime-Definitionen.

Alle Regime teilen denselben Parametersatz (gemeinsamer Kern). Ein Regime
überschreibt ausschließlich die Parameter im Block "Institutionen".
Damit gehen Unterschiede in den Ergebnissen nur auf die Institutionen zurück.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields
from typing import Optional


@dataclass(frozen=True)
class Params:
    # --- Simulation -------------------------------------------------------
    n_households: int = 300
    n_firms: int = 30
    periods: int = 600
    burn_in: int = 200            # Einschwingphase, wird nicht ausgewertet

    # --- Haushalte ----------------------------------------------------------
    init_money_hh: float = 10.0   # mittleres Startvermögen
    wealth_sigma: float = 0.6     # Streuung des Startvermögens (lognormal)
    c_income: float = 0.8         # Konsumneigung aus Einkommen
    c_wealth: float = 0.05        # Konsumneigung aus Vermögen
    search_goods: int = 3         # besuchte Anbieter pro Periode
    search_jobs: int = 4          # Bewerbungen pro Periode
    memory: int = 50              # so lange wird ein Betrüger gemieden

    # --- Unternehmen --------------------------------------------------------
    productivity: float = 1.0
    init_money_firm: float = 20.0
    init_price: float = 1.0
    init_wage: float = 0.9
    init_utilization: float = 0.85  # Anteil der Haushalte, die zu Beginn beschäftigt sind
    init_markup: float = 0.1      # Anfangsaufschlag auf Stückkosten
    markup_step: float = 0.01     # max. Änderung des Aufschlags pro Periode
    qty_adjust: float = 0.3       # Anpassungsgeschwindigkeit des Produktionsziels
    lost_weight: float = 0.5      # Gewicht abgewiesener Nachfrage (Kunden kaufen oft woanders)
    inventory_buffer: float = 0.05  # gewünschter Lagerpuffer über erwarteter Nachfrage
    wage_step: float = 0.02       # max. relative Lohnänderung pro Periode
    wage_indexation: float = 0.0  # Inflationsausgleich der Löhne (Anteil der Vorperioden-Inflation)
    wage_cut_threshold: float = 0.05  # Löhne sinken nur, wenn die Arbeitslosigkeit darüber liegt
    min_markup: float = 0.0       # Preisuntergrenze = Stückkosten * (1 + min_markup)
    inventory_decay: float = 0.5  # Verderb unverkaufter Ware
    dividend_share: float = 0.5   # ausgeschütteter Gewinnanteil
    cash_reserve: float = 4.0     # Liquiditätsreserve in geplanten Lohnsummen; Überschuss wird ausgeschüttet
    entry_capital: float = 20.0   # Startkapital einer neuen Firma (in Anfangslöhnen, mitwachsend)

    # --- Verhalten ohne Durchsetzung / ohne Wettbewerbsaufsicht -------------
    dishonest_share: float = 0.3  # Anteil der Firmen, die bereit sind zu betrügen
    fraud_prob: float = 0.3       # Betrugswahrscheinlichkeit je Transaktion (ohne Durchsetzung)
    fraud_loss: float = 0.5       # vorenthaltener Anteil (Ware bzw. Lohn)
    enforcement_effect: float = 0.9  # Senkung der Betrugsrate durch Vertragsdurchsetzung
    market_power: float = 1.0     # Zusatzaufschlag = market_power * Marktanteil (ohne Kartellaufsicht)
    entry_prob_free: float = 0.1  # Markteintrittswahrscheinlichkeit je Periode ohne Wettbewerbsaufsicht

    # --- Kreditmarkt / Bank ------------------------------------------------
    credit: bool = True               # Kreditmarkt an/aus (für Vergleiche ohne Kredit)
    bank_equity_init: float = 0.1     # Anfangseigenkapital der Bank relativ zur Anfangsgeldmenge
    bank_owner_share: float = 0.1     # Anteil der Haushalte, die Bankanteile halten
    bank_dividend_share: float = 0.5  # ausgeschütteter Anteil des Bankgewinns
    base_rate: float = 0.004          # Grundzins je Periode (≈ 5 % p. a.)
    risk_premium: float = 0.01        # Zinsaufschlag je Einheit Verschuldungsgrad (je Periode)
    max_leverage: float = 0.7         # max. Schulden / Aktiva einer Firma für Neukredite
    loan_maturity: int = 24           # Laufzeit (Perioden); Tilgung in gleichen Raten
    loan_buffer: float = 1.0          # zusätzlich beantragte Lohnsummen als Liquiditätspuffer
    cash_reserve_credit: float = 1.0  # Liquiditätsreserve (in Lohnsummen), wenn Kredit verfügbar ist
    rationing_memory: int = 12        # nach einer Kreditablehnung so lange wieder volle Reserve
    recovery_free: float = 0.3        # Verwertungsquote bei Insolvenz ohne Rechtsdurchsetzung
    bank_min_equity: float = 0.02     # Bankeigentümer halten mind. diesen Anteil der Geldmenge als Eigenkapital
    bank_recap_share: float = 0.1     # max. Anteil ihres Geldes, den Eigentümer je Periode nachschießen

    # --- Externer Schock (in allen Regimen identisch) -----------------------
    shock_period: int = 400
    shock_duration: int = 30
    shock_size: float = 0.25      # Produktivitätseinbruch

    # --- Institutionen (werden von den Regimen gesetzt) ---------------------
    property_rights: bool = True        # Verträge werden durchgesetzt
    antitrust: bool = False             # Wettbewerbsaufsicht, freier Markteintritt
    tax_rate: float = 0.0               # Steuer auf Löhne und Dividenden
    benefit_ratio: float = 0.0          # Arbeitslosengeld relativ zum Nettodurchschnittslohn
    public_sector_share: float = 0.0    # Anteil der Haushalte im Staatsdienst (Verwaltungskosten)
    reservation_factor: float = 1.0     # Job wird angenommen, wenn Nettolohn >= Faktor * Leistung
    price_cap: Optional[float] = None   # max. Aufschlag auf Stückkosten (C)
    min_wage_ratio: Optional[float] = None  # Mindestlohn relativ zu Preisniveau * Produktivität (C)
    min_wage_indexation: int = 12           # Anpassungsintervall des Mindestlohns (Perioden)
    capital_requirement: float = 0.02   # Mindest-Eigenkapitalquote der Bank (A: selbst gewählt)
    deposit_insurance: bool = False     # Staat rekapitalisiert die Bank bei Verlusten (B/C)
    interest_cap: Optional[float] = None  # Zinsobergrenze je Periode (C)
    large_exposure_limit: Optional[float] = None  # max. Kredit je Firma relativ zum Bankkapital (B/C)


# Gemeinsamer Staatsrahmen von B (C baut darauf auf)
_RAHMEN = dict(
    property_rights=True,
    antitrust=True,
    tax_rate=0.2,
    benefit_ratio=0.5,
    public_sector_share=0.05,
    reservation_factor=1.1,
    capital_requirement=0.08,
    deposit_insurance=True,
    large_exposure_limit=0.25,
)

REGIMES: dict[str, dict] = {
    # A ohne jede Vertragsdurchsetzung: nur Reputation schützt (Untergrenze für A)
    "A_frei": dict(property_rights=False, antitrust=False),
    # A mit kostenlos gesicherten Eigentumsrechten, sonst kein Staat (Obergrenze für A)
    "A_eigentum": dict(property_rights=True, antitrust=False),
    # B: Staat setzt Regeln (Recht, Wettbewerb, Steuern/Grundsicherung)
    "B_rahmen": dict(_RAHMEN),
    # C: wie B plus direkte Eingriffe in Preise und Löhne, größere Verwaltung
    "C_eingriff": dict(_RAHMEN, public_sector_share=0.07, price_cap=0.05, min_wage_ratio=0.8,
                       interest_cap=0.006),
}

REGIME_LABELS = {
    "A_frei": "A frei (ohne Rechtsdurchsetzung)",
    "A_eigentum": "A frei (mit Eigentumsrechten)",
    "B_rahmen": "B Regelrahmen",
    "C_eingriff": "C Direkte Eingriffe",
}


_FIELD_NAMES = {f.name for f in fields(Params)}


def make_params(regime: str, base: Params | None = None, **overrides) -> Params:
    """Erzeugt den Parametersatz eines Regimes. overrides gelten zuletzt (für Sweeps)."""
    unknown = set(overrides) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"Unbekannte Parameter: {sorted(unknown)}")
    p = base or Params()
    p = replace(p, **REGIMES[regime])
    return replace(p, **overrides)

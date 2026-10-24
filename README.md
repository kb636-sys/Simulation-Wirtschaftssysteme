# Wirtschaftssimulation: Vergleich dreier Wirtschaftssysteme

Agentenbasierter Prototyp, der drei institutionelle Ordnungen auf **demselben Modellkern** vergleicht:

| Regime | Beschreibung |
|---|---|
| **A frei (ohne Rechtsdurchsetzung)** | Kein Staat. Verträge – auch Kreditverträge – werden nicht durchgesetzt; nur Reputation schützt vor Betrug. Bank wählt ihr Eigenkapital selbst (2 %), keine Einlagensicherung. Untergrenze für A. |
| **A frei (mit Eigentumsrechten)** | Kein Staat, aber Eigentum und Verträge gelten als (kostenlos) gesichert. Bankregeln wie oben. Obergrenze für A. |
| **B Regelrahmen** | Staat setzt Regeln: Vertragsdurchsetzung, Wettbewerbsaufsicht (keine Marktmacht, freier Marktzutritt), 20 % Steuer, Arbeitslosengeld, 5 % der Erwerbspersonen im Staatsdienst; Bankenaufsicht mit 8 % Eigenkapital, Großkreditgrenze (25 % des Bankkapitals je Firma) und Einlagensicherung. |
| **C Direkte Eingriffe** | Wie B, zusätzlich Preisdeckel (max. 5 % Aufschlag auf Stückkosten), Mindestlohn (80 % des Wertgrenzprodukts, jährlich indexiert), Zinsobergrenze (0,6 % je Periode ≈ 7,4 % p. a.), größere Verwaltung (7 %). |

A wird in zwei Varianten gerechnet, weil „vollkommen frei“ zwei sinnvolle Lesarten hat. Beide sind in `config.py` einstellbar, ebenso jeder einzelne Baustein von B und C.

## Schnellstart

```bash
pip install numpy pandas matplotlib
python run_experiments.py --runs 30          # Hauptexperiment (≈ 4 s je Lauf und Kern)
python run_experiments.py --sweep-only --sweep-runs 8   # Sensitivitätsläufe (abschnittsweise gespeichert, fortsetzbar)
python analysis.py                           # Diagramme nach figures/, Tabelle ERGEBNISSE.md
```

Einen einzelnen Lauf interaktiv ansehen:

```python
from config import make_params
from model import Economy
ts = Economy(make_params("C_eingriff", price_cap=0.1), seed=1).run()
ts.plot(x="t", y=["konsum_pk", "arbeitslosigkeit"])
```

## Dateien

| Datei | Inhalt |
|---|---|
| `config.py` | Alle Parameter (`Params`) und die Regime-Definitionen (`REGIMES`) |
| `model.py` | Das Modell (`Economy`): Agenten, Märkte, Staat, Kennzahlen je Periode |
| `metrics.py` | Gini, HHI, Lorenzkurve, Zusammenfassung eines Laufs, Kennzahlenkatalog |
| `run_experiments.py` | Batch-Läufe mit festen Seeds, parallelisiert; Sensitivitätsläufe (`SWEEPS`) |
| `analysis.py` | Diagramme und Ergebnistabelle |
| `results/` | Rohdaten: Zeitreihen, Kennzahlen je Lauf, Lorenzkurven, Sensitivität |
| `figures/` | Erzeugte Diagramme |

## Modellbeschreibung (ODD-Kurzform)

**Zweck.** Vergleich von Effizienz, Verteilung, Stabilität und Marktfunktion unter drei institutionellen Ordnungen bei identischer Bevölkerung, Technologie und identischen Schocks.

**Einheiten.** 300 Haushalte, 30 Firmenplätze, eine Bank (gehört 10 % der Haushalte), Staat (nur B/C). Eine Periode ≈ ein Monat; 600 Perioden.

**Ablauf einer Periode.**
1. *Löhne:* Firmen mit unbesetzten Stellen erhöhen den Lohn; bei Arbeitslosigkeit über 5 % senken die übrigen leicht. In C gilt der Mindestlohn.
2. *Preise:* Preis = Stückkosten × (1 + Aufschlag). Ausverkauft → Aufschlag steigt, Ware übrig → Aufschlag sinkt. Ohne Kartellaufsicht hebt Marktmacht den Mindestaufschlag (proportional zum Marktanteil). In C deckelt der Preisdeckel den Aufschlag.
3. *Produktionsziel:* geglättete Nachfrage (Verkäufe + abgewiesene Kunden) plus kleiner Lagerpuffer.
4. *Arbeitsnachfrage und Kredit:* Reicht die Kasse nicht für die geplante Belegschaft, beantragt die Firma einen Kredit (siehe Kreditmarkt). Was dann noch nicht finanziert ist, wird nicht eingestellt; überzählige Beschäftigte werden entlassen.
5. *Arbeitsmarkt:* Arbeitslose bewerben sich bei einigen Firmen und nehmen den höchsten Lohn, sofern er über dem Reservationslohn (Bezug zum Arbeitslosengeld) liegt.
6. *Lohnzahlung und Produktion.* Unehrliche Firmen können Lohn vorenthalten.
7. *Staat (B/C):* Steuern auf Löhne und Dividenden; daraus Staatslöhne, dann Arbeitslosengeld; Überschuss wird pauschal zurückgegeben.
8. *Gütermarkt:* Haushalte budgetieren aus Einkommen und Vermögen, besuchen einige Anbieter und kaufen beim günstigsten. Unehrliche Firmen können weniger liefern als bezahlt.
9. *Schuldendienst und Gewinne:* Firmen zahlen Zins und Tilgung. Kasse oberhalb einer Liquiditätsreserve tilgt zuerst Schulden, der Rest geht mit dem Gewinnanteil an den Eigentümer.
10. *Insolvenz und Markteintritt:* Firmen, die keinen Lohn mehr zahlen können oder überschuldet sind, scheiden aus; die Bank verwertet die Kasse und schreibt den Rest ab. Neue Firmen werden von Haushalten gegründet, die die Hälfte des Startkapitals einbringen; den Rest kann die Bank finanzieren. Mit Wettbewerbsaufsicht ist der Eintritt sofort möglich, ohne nur mit 10 % Wahrscheinlichkeit je Periode.
11. *Bank:* Zinsertrag minus Abschreibungen ergibt den Bankgewinn. Wird Eigenkapital knapp, schießen die Eigentümer nach; reicht das nicht, rettet in B/C der Staat (Einlagensicherung), in A tragen die Einleger den Verlust (anteilige Kürzung der Guthaben).

**Wichtige Designkonzepte.**
- *Betrug und Reputation:* Ohne Rechtsdurchsetzung betrügen unehrliche Firmen (30 % der Firmen) in 30 % der Transaktionen und behalten die Hälfte ein. Betrogene meiden die Firma 50 Perioden lang. Mit Rechtsdurchsetzung sinkt die Betrugsrate um 90 %.
- *Geld und Bankbilanz:* Kredite schaffen Buchgeld, Tilgungen vernichten es. Die Bilanzidentität *Eigenkapital = Reserven + Kredite − Einlagen* wird in jeder Periode als `bilanzfehler` mitgeschrieben und ist exakt 0. Ohne Kreditmarkt (`credit=False`) bleibt die Geldmenge konstant.

### Kreditmarkt

| Baustein | Regel |
|---|---|
| Kreditnachfrage | Fehlbetrag für die geplante Lohnsumme plus eine Lohnsumme Puffer. Mit Kreditzugang halten Firmen nur 1 Lohnsumme Reserve statt 4; wer abgelehnt wurde, hält 12 Perioden lang wieder die volle Reserve. |
| Kreditprüfung | Abgelehnt, wenn Schulden/Aktiva nach dem Kredit über 70 % lägen, wenn der Zins über einer Zinsobergrenze läge (C) oder wenn die Firma einen Kredit schon einmal absichtlich nicht bedient hat. |
| Zins | Grundzins 0,4 % je Periode (≈ 5 % p. a.) + Risikoaufschlag proportional zum Verschuldungsgrad; ohne Rechtsdurchsetzung höher, weil die Bank bei Insolvenz weniger zurückbekommt. |
| Tilgung | Laufzeit 24 Perioden, gleiche Raten. |
| Ausfall | Mit Rechtsdurchsetzung verwertet die Bank die gesamte Kasse, ohne nur 30 % (Rest „verschwindet“ zum Eigentümer). Unehrliche Firmen stellen ohne Durchsetzung zeitweise die Zahlungen absichtlich ein. |
| Bankkapital | Kreditvolumen ≤ Eigenkapital / Mindestquote (A: 2 %, B/C: 8 %). B/C zusätzlich Großkreditgrenze 25 %. |
| Krise | Eigenkapital < 0. Private Nachschüsse → staatliche Rettung (B/C) → Verlust der Einleger (Bail-in). |
- *Schock:* Produktivitätseinbruch um 25 % in den Perioden 400–429, in allen Regimen identisch.

## Wie die Vergleichbarkeit gesichert wird

- **Ein Kern, ein Parametersatz.** Regime überschreiben nur die Parameter im Block „Institutionen“ (`make_params`).
- **Common Random Numbers.** Seed *s* erzeugt in allen Regimen dieselbe Startbevölkerung (Vermögen, Firmeneigentum, ehrliche/unehrliche Firmen, Startbeschäftigung) über einen eigenen Zufallsstrom. Die Ergebnistabelle nutzt das für **gepaarte Differenzen** (Regime − B je Seed), die deutlich genauer sind als ein Vergleich der Mittelwerte.
- **Einschwingphase** (200 Perioden) wird verworfen; Niveaus werden in der ruhigen Phase 200–399 gemessen, Schockkennzahlen danach.
- **Viele Läufe** mit 95-%-Konfidenzintervallen.
- **Regler statt Schalter:** Preisdeckel, Mindestlohn und Steuersatz werden über ihren Bereich variiert.
- **Robustheitsprüfung:** Marktmacht, Betrugsneigung, Konsumverhalten und Preisanpassungstempo werden für alle Regime variiert. Eine Aussage gilt nur als belastbar, wenn die Rangfolge über die Achse hält.
- **Kein Gesamtsieger:** Die Kennzahlen decken verschiedene Ziele ab; ihre Gewichtung ist normativ.

## Hinweise zu den ersten Ergebnissen

- **Kreditmarkt.** Ohne Rechtsdurchsetzung ist Kredit teurer, Ausfälle sind fast dreimal so hoch, und es kommt zu Bankenkrisen mit Einlegerverlusten. In C wirkt die Zinsobergrenze faktisch als Kreditsperre (99 % Ablehnungen). Die Eigenkapitalanforderung ist im untersuchten Bereich (2–16 %) in A mit Eigentumsrechten, B und C nicht bindend.
- **Schockreaktion in C** hängt vor allem am preisindexierten Mindestlohn: Bei `min_wage_ratio <= 0.7` ist der Konsumverlust nicht größer als in B.

- **Wechselwirkung Mindestlohn × Preisdeckel in C.** Der Mindestlohn ist an das Preisniveau gekoppelt. Liegt der erlaubte Aufschlag über ca. 25 %, ist der Mindestlohn (80 % von Preis × Produktivität) höher als der Lohn, den die Preise tragen: Bei jeder Indexierung steigen Löhne, dann Preise, dann wieder der Mindestlohn. Bei fester Geldmenge endet das in Massenarbeitslosigkeit (Sensitivitätslauf „Preisdeckel 0,4“). Derselbe Mechanismus erzeugt die Sägezahn-Einbrüche von C während des Produktivitätsschocks.
- **A ohne Rechtsdurchsetzung** erreicht bis Periode 600 kein Gleichgewicht: Ungleichheit und Arbeitslosigkeit steigen weiter. Für belastbare Niveaus längere Läufe rechnen (`periods`).
- **Mindestlohn bis 0,85** ist bei den Standardparametern kaum bindend; Effekte zeigen sich erst darüber.

## Grenzen des Prototyps

Die Ergebnisse folgen aus den Annahmen – sie sind kein Beleg für oder gegen ein reales Wirtschaftssystem. Insbesondere:
- Ein einziges homogenes Gut, keine Investitionen, kein Kapitalstock, kein Wachstum. Kredite finanzieren nur Betriebsmittel (Löhne), keine Investitionen; Haushalte nehmen keine Kredite auf.
- Eine einzige Bank ohne Wettbewerb, ohne Zentralbank und ohne Einlagenzins.
- Arbeitsangebot ist unelastisch; Steuern verzerren nur über den Reservationslohn.
- Der Staat ist ein einfacher Regelautomat ohne Politikökonomie (keine Lobbyarbeit, keine Fehlanreize der Verwaltung, keine Schattenwirtschaft bei Preisdeckeln).
- Marktmacht, Betrug und Markteintritt sind stilisierte Mechanismen, deren Stärke maßgeblich über Parameter gesteuert wird.

Naheliegende Erweiterungen: mehrere Banken und Interbankenmarkt, Zentralbank mit Leitzins, Investitionen und Produktivitätswachstum (Innovation), Schwarzmärkte bei Preisdeckeln, Steuerhinterziehung, Kalibrierung an empirischen Daten (z. B. Arbeitslosigkeit, Lohnquote, Gini).

Orientierung für die Modellbausteine: Delli Gatti et al. (2011) *Macroeconomics from the Bottom-up*; Dosi, Fagiolo & Roventini (2010) *Schumpeter meeting Keynes*; Grimm et al. (2020) zum ODD-Protokoll.

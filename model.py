"""Agentenbasiertes Modell einer geschlossenen Geldwirtschaft.

Gemeinsamer Kern für alle Regime:
  * Haushalte arbeiten, konsumieren und besitzen Firmen (Dividenden).
  * Firmen setzen Löhne, Preise und Produktionsmengen nach einfachen
    adaptiven Regeln (angelehnt an Delli Gatti et al. 2011, "Macroeconomics
    from the Bottom-up").
  * Arbeits- und Gütermarkt mit begrenzter Suche (dezentrale Zuordnung).
  * Eine Bank vergibt Kredite an Firmen und schafft dabei Geld (Buchgeld);
    Tilgung vernichtet es wieder. Es gilt stets die Bilanzidentität
        Eigenkapital Bank = Reserven + Kredite − Einlagen (= Geld aller Agenten),
    die in jeder Periode als `bilanzfehler` mitgeschrieben wird.

Die Institutionen (Regime A, B, C) wirken ausschließlich über die Parameter
im Block "Institutionen" von config.Params.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import Params
from metrics import gini, hhi

EPS = 1e-9
UNEMPLOYED, PUBLIC = -1, -2


class Economy:
    def __init__(self, p: Params, seed: int):
        self.p = p
        # Getrennte Zufallsströme ("Common Random Numbers"):
        # Startbedingungen sind bei gleichem Seed in allen Regimen identisch.
        s_init, s_dyn = np.random.SeedSequence(seed).spawn(2)
        r = np.random.default_rng(s_init)
        self.rng = np.random.default_rng(s_dyn)

        H, F = p.n_households, p.n_firms
        mu = math.log(p.init_money_hh) - p.wealth_sigma ** 2 / 2
        self.h_money = r.lognormal(mu, p.wealth_sigma, H)
        self.h_employer = np.full(H, UNEMPLOYED, dtype=int)
        self.h_div_last = np.zeros(H)
        self.h_blacklist: list[dict[int, int]] = [dict() for _ in range(H)]

        self.f_alive = np.ones(F, dtype=bool)
        self.f_uid = np.arange(F)
        self.next_uid = F
        self.f_money = np.full(F, p.init_money_firm)
        self.f_price = p.init_price * (1 + r.uniform(-0.05, 0.05, F))
        self.f_wage = np.full(F, p.init_wage)
        self.f_target = np.full(F, p.init_utilization * H / F * p.productivity)
        self.f_inventory = np.zeros(F)
        self.f_prod_last = self.f_target.copy()
        self.f_sales = self.f_target.copy()
        self.f_stockout = np.zeros(F, dtype=bool)
        self.f_lost = np.zeros(F)
        self.f_markup = np.full(F, p.init_markup)
        self.f_unfilled = np.zeros(F, dtype=bool)
        self.f_honest = r.random(F) >= p.dishonest_share
        self.f_owner = r.integers(0, H, F)
        self.f_debt = np.zeros(F)
        self.f_last_rationed = np.full(F, -10**6)  # Periode der letzten Kreditablehnung
        self.f_rate = np.zeros(F)
        self.bank_blocked: set[int] = set()   # Firmen, die den Kredit absichtlich nicht bedient haben
        n_owners = max(1, int(p.bank_owner_share * H))
        self.bank_owners = r.choice(H, n_owners, replace=False)

        employed = r.permutation(H)[: int(p.init_utilization * H)]
        self.h_employer[employed] = np.arange(len(employed)) % F

        self.treasury = 0.0
        self.benefit_last = 0.0
        self.min_wage = 0.0
        self.inflation_last = 0.0
        self.unemployment_last = 0.0
        self.price_index = p.init_price
        self.avg_wage = p.init_wage
        self.t = 0
        self.records: list[dict] = []

        # Bank: Reserven bleiben konstant, Eigenkapital ergibt sich aus der Bilanz
        m0 = self.money_total()
        self.bank_equity = p.bank_equity_init * m0
        self.reserves = m0 + self.bank_equity

    # ------------------------------------------------------------------ helpers
    def _trusted(self, h: int, firms: np.ndarray) -> np.ndarray:
        bl = self.h_blacklist[h]
        if not bl:
            return firms
        t = self.t
        return np.array([f for f in firms if bl.get(self.f_uid[f], -1) < t], dtype=int)

    def _cheat(self, f: int, p_cheat: float) -> bool:
        return (not self.f_honest[f]) and p_cheat > 0 and self.rng.random() < p_cheat

    def _blacklist(self, h: int, f: int):
        bl = self.h_blacklist[h]
        bl[self.f_uid[f]] = self.t + self.p.memory
        if len(bl) > 20:  # abgelaufene Einträge aufräumen
            for k in [k for k, v in bl.items() if v < self.t]:
                del bl[k]

    def wealth(self) -> np.ndarray:
        """Vermögen = Geld + Nettokasse eigener Firmen + Anteil am Bankkapital."""
        alive = self.f_alive
        net = self.f_money[alive] - self.f_debt[alive]
        owned = np.bincount(self.f_owner[alive], weights=net, minlength=self.p.n_households)
        w = self.h_money + owned
        if self.bank_equity > 0:
            w[self.bank_owners] += self.bank_equity / len(self.bank_owners)
        return w

    def loans_total(self) -> float:
        return float(self.f_debt[self.f_alive].sum())

    def balance_error(self) -> float:
        """Abweichung von: Eigenkapital = Reserven + Kredite − Einlagen (sollte 0 sein)."""
        return self.bank_equity - (self.reserves + self.loans_total() - self.money_total())

    def _loan(self, f: int, amount: float) -> float:
        """Kreditantrag einer Firma. Gibt den gewährten Betrag zurück."""
        p = self.p
        if amount <= EPS:
            return 0.0
        self.c_requests += 1
        if self.f_uid[f] in self.bank_blocked:
            self.c_rationed += 1
            self.f_last_rationed[f] = self.t
            return 0.0
        # Kapazität der Bank (Eigenkapitalanforderung)
        capacity = self.bank_equity / p.capital_requirement - self.loans_total() \
            if p.capital_requirement > 0 else np.inf
        amount = min(amount, max(capacity, 0.0))
        if p.large_exposure_limit is not None:  # Großkreditgrenze
            amount = min(amount, max(p.large_exposure_limit * self.bank_equity - self.f_debt[f], 0.0))
        assets = self.f_money[f] + amount + self.f_inventory[f] * self.f_price[f]
        lev = (self.f_debt[f] + amount) / max(assets, EPS)
        recovery = 1.0 if p.property_rights else p.recovery_free
        rate = p.base_rate + p.risk_premium * lev * (2 - recovery)  # schlechtere Verwertung -> teurer
        if (amount < 0.25 * self.f_wage[f] or lev > p.max_leverage
                or (p.interest_cap is not None and rate > p.interest_cap)):
            self.c_rationed += 1
            self.f_last_rationed[f] = self.t
            return 0.0
        d = self.f_debt[f]
        self.f_rate[f] = (self.f_rate[f] * d + rate * amount) / (d + amount)
        self.f_debt[f] += amount
        self.f_money[f] += amount
        self.c_new += amount
        return amount

    def money_total(self) -> float:
        return float(self.h_money.sum() + self.f_money[self.f_alive].sum() + self.treasury)

    # ---------------------------------------------------------------- one period
    def step(self):
        p, rng, t = self.p, self.rng, self.t
        H, F = p.n_households, p.n_firms
        in_shock = p.shock_period <= t < p.shock_period + p.shock_duration
        prod = p.productivity * ((1 - p.shock_size) if in_shock else 1.0)
        p_cheat = p.fraud_prob * ((1 - p.enforcement_effect) if p.property_rights else 1.0)
        emp = self.h_employer
        alive = np.flatnonzero(self.f_alive)
        n = len(alive)
        self.c_requests = self.c_rationed = 0
        self.c_new = 0.0
        loans_start = self.loans_total()

        # 1. Löhne: Erhöhung bei unbesetzten Stellen, Senkung nur bei hoher
        #    Arbeitslosigkeit; optional Inflationsausgleich; ggf. Mindestlohn
        u = rng.uniform(0, p.wage_step, n)
        w = self.f_wage[alive] * (1 + p.wage_indexation * self.inflation_last)
        slack = self.unemployment_last > p.wage_cut_threshold
        w = np.where(self.f_unfilled[alive], w * (1 + u), w * (1 - 0.5 * u) if slack else w)
        if p.min_wage_ratio is not None:
            # Mindestlohn wird nur alle min_wage_indexation Perioden an das Preisniveau angepasst
            if t % p.min_wage_indexation == 0:
                self.min_wage = p.min_wage_ratio * self.price_index * p.productivity
            w = np.maximum(w, self.min_wage)
        self.f_wage[alive] = w

        # 2. Aufschlag (Markup) und Produktionsziel
        #    Ausverkauft -> Aufschlag steigt; Ware übrig -> Aufschlag sinkt.
        #    Preis = Stückkosten * (1 + Aufschlag). Ohne Kartellaufsicht erhöht
        #    Marktmacht den Mindestaufschlag; ein Preisdeckel begrenzt ihn nach oben.
        tot_sales = self.f_sales[alive].sum()
        share = self.f_sales[alive] / tot_sales if tot_sales > 0 else np.zeros(n)
        power = 0.0 if p.antitrust else p.market_power * share
        m = self.f_markup[alive]
        dm = rng.uniform(0, p.markup_step, n)
        so = self.f_stockout[alive]
        leftover = self.f_inventory[alive] > 0.05 * np.maximum(self.f_prod_last[alive], 1.0)
        m = np.where(so, m + dm, np.where(leftover, m - dm, m))
        m = np.maximum(m, p.min_markup + power)
        if p.price_cap is not None:
            m = np.minimum(m, p.price_cap)
        self.f_markup[alive] = m
        self.f_price[alive] = w / prod * (1 + m)
        # Erwartete Nachfrage = Verkäufe + abgewiesene Nachfrage (geglättet)
        demand_est = (self.f_sales[alive] + p.lost_weight * self.f_lost[alive]) * (1 + p.inventory_buffer)
        target = self.f_target[alive]
        target = target + p.qty_adjust * (demand_est - target)
        self.f_target[alive] = np.maximum(target, 1.0)

        # 3. Arbeitsnachfrage (begrenzt durch Kasse; Lücke ggf. per Kredit)
        vac = np.zeros(F, dtype=int)
        for f in alive:
            need = math.ceil(max(self.f_target[f] - self.f_inventory[f], 0) / prod)
            afford = int(self.f_money[f] // self.f_wage[f])
            if p.credit and need > afford:
                gap = (need + p.loan_buffer) * self.f_wage[f] - self.f_money[f]
                self._loan(f, gap)
                afford = int(self.f_money[f] // self.f_wage[f])
            demand = min(need, afford)
            workers = np.flatnonzero(emp == f)
            if len(workers) > demand:
                emp[rng.choice(workers, len(workers) - demand, replace=False)] = UNEMPLOYED
            else:
                vac[f] = demand - len(workers)

        # 4. Staat stellt ein (Verwaltung, Rechtsdurchsetzung)
        pub_target = int(round(p.public_sector_share * H))
        pub = np.flatnonzero(emp == PUBLIC)
        if len(pub) < pub_target:
            pool = np.flatnonzero(emp == UNEMPLOYED)
            k = min(pub_target - len(pub), len(pool))
            if k:
                emp[rng.choice(pool, k, replace=False)] = PUBLIC
        elif len(pub) > pub_target:
            emp[rng.choice(pub, len(pub) - pub_target, replace=False)] = UNEMPLOYED

        # 5. Arbeitsmarkt: Arbeitslose bewerben sich, nehmen das beste Angebot
        reservation = p.reservation_factor * self.benefit_last
        for h in rng.permutation(np.flatnonzero(emp == UNEMPLOYED)):
            open_f = self._trusted(h, np.flatnonzero(vac > 0))
            if len(open_f) == 0:
                continue
            k = min(p.search_jobs, len(open_f))
            sample = open_f if k == len(open_f) else rng.choice(open_f, k, replace=False)
            best = sample[np.argmax(self.f_wage[sample])]
            if self.f_wage[best] * (1 - p.tax_rate) >= reservation:
                emp[h] = best
                vac[best] -= 1
        self.f_unfilled[:] = False
        self.f_unfilled[alive] = vac[alive] > 0

        # 6. Lohnzahlung (mit möglichem Lohnbetrug), Produktion
        income = np.zeros(H)
        wage_bill = np.zeros(F)
        fraud_value, contract_value = 0.0, 0.0
        quitters = []
        output = 0.0
        for f in alive:
            ws = np.flatnonzero(emp == f)
            if len(ws) == 0:
                continue
            pay = np.full(len(ws), self.f_wage[f])
            contract_value += pay.sum()
            if not self.f_honest[f] and p_cheat > 0:
                cheat = rng.random(len(ws)) < p_cheat
                if cheat.any():
                    fraud_value += pay[cheat].sum() * p.fraud_loss
                    pay[cheat] *= 1 - p.fraud_loss
                    for h in ws[cheat]:
                        self._blacklist(h, f)
                        quitters.append(h)
            self.f_money[f] -= pay.sum()
            wage_bill[f] = pay.sum()
            income[ws] += pay
            q = len(ws) * prod
            self.f_inventory[f] += q
            output += q
        self.f_prod_last[:] = 0.0
        for f in alive:
            self.f_prod_last[f] = (emp == f).sum() * prod

        # 7. Staat: Steuern, Staatslöhne, Arbeitslosengeld, Überschussausschüttung
        tax = p.tax_rate * income
        income -= tax
        self.treasury += tax.sum()
        self.h_money += income  # Nettolöhne der Firmen-Beschäftigten
        pub = np.flatnonzero(emp == PUBLIC)
        net_pub = self.avg_wage * (1 - p.tax_rate)
        if len(pub):
            per = min(net_pub, self.treasury / len(pub))
            self.h_money[pub] += per
            income[pub] += per
            self.treasury -= per * len(pub)
        unemp = np.flatnonzero(emp == UNEMPLOYED)
        target_b = p.benefit_ratio * self.avg_wage * (1 - p.tax_rate)
        self.benefit_last = 0.0
        if target_b > 0 and len(unemp):
            b = min(target_b, self.treasury / len(unemp))
            self.h_money[unemp] += b
            income[unemp] += b
            self.treasury -= b * len(unemp)
            self.benefit_last = b
        buffer = 5 * (len(pub) * net_pub + target_b * 0.1 * H)
        if self.treasury > buffer:  # Überschuss wird pauschal zurückgegeben
            extra = (self.treasury - buffer) / H
            self.h_money += extra
            income += extra
            self.treasury = buffer
        if quitters:
            emp[np.array(quitters)] = UNEMPLOYED

        # 8. Gütermarkt: Haushalte besuchen wenige Anbieter, kaufen beim günstigsten
        spend_income = income + self.h_div_last
        budgets = np.minimum(self.h_money, p.c_income * spend_income + p.c_wealth * self.h_money)
        consumption = np.zeros(H)
        revenue, sold, charged = np.zeros(F), np.zeros(F), np.zeros(F)
        self.f_stockout[:] = False
        budget_total, unmet = 0.0, 0.0
        lost = np.zeros(F)
        for h in rng.permutation(H):
            rem = budgets[h]
            if rem <= EPS:
                continue
            budget_total += rem
            cands = self._trusted(h, alive)
            if len(cands):
                k = min(p.search_goods, len(cands))
                sample = rng.choice(cands, k, replace=False)
                for f in sample[np.argsort(self.f_price[sample])]:
                    if self.f_inventory[f] <= EPS:
                        self.f_stockout[f] = True
                        lost[f] += rem / self.f_price[f]
                        continue
                    pr = self.f_price[f]
                    q = min(rem / pr, self.f_inventory[f])
                    delivered = q
                    if self._cheat(f, p_cheat):
                        delivered = q * (1 - p.fraud_loss)
                        fraud_value += (q - delivered) * pr
                        self._blacklist(h, f)
                    contract_value += q * pr
                    self.f_inventory[f] -= delivered
                    if self.f_inventory[f] <= EPS:
                        self.f_inventory[f] = 0.0
                        self.f_stockout[f] = True
                    cost = q * pr
                    self.h_money[h] -= cost
                    self.f_money[f] += cost
                    revenue[f] += cost
                    sold[f] += delivered
                    charged[f] += q
                    consumption[h] += delivered
                    rem -= cost
                    if rem <= EPS:
                        rem = 0.0
                        break
            unmet += rem

        # 9a. Schuldendienst: Zins + Tilgung (ohne Rechtsdurchsetzung auch absichtlicher Ausfall)
        interest_paid = np.zeros(F)
        for f in alive:
            debt = self.f_debt[f]
            if debt <= EPS:
                continue
            interest = debt * self.f_rate[f]
            due = interest + debt / p.loan_maturity
            contract_value += due
            if self._cheat(f, p_cheat):
                fraud_value += due
                self.bank_blocked.add(int(self.f_uid[f]))
                continue
            pay = min(due, max(self.f_money[f], 0.0))
            pi = min(pay, interest)
            self.f_money[f] -= pay
            self.f_debt[f] -= pay - pi
            interest_paid[f] = pi
        interest_income = float(interest_paid.sum())

        # 9b. Gewinne und Dividenden (Dividenden fließen in den Konsum der Folgeperiode)
        div = np.zeros(H)
        for f in alive:
            profit = revenue[f] - wage_bill[f] - interest_paid[f]
            # Kasse oberhalb der Liquiditätsreserve: erst Schulden tilgen, dann ausschütten
            planned = self.f_wage[f] * math.ceil(self.f_target[f] / p.productivity)
            # Wer kürzlich keinen Kredit bekam, hält wieder mehr Kasse
            credit_ok = p.credit and t - self.f_last_rationed[f] > p.rationing_memory
            reserve = p.cash_reserve_credit if credit_ok else p.cash_reserve
            excess = self.f_money[f] - reserve * planned
            if excess > 0 and self.f_debt[f] > 0:
                pre = min(excess, self.f_debt[f])
                self.f_money[f] -= pre
                self.f_debt[f] -= pre
                excess -= pre
            d = p.dividend_share * max(profit, 0.0) + max(excess, 0.0)
            d = min(d, max(self.f_money[f], 0.0))
            if d > 0:
                self.f_money[f] -= d
                tx = d * p.tax_rate
                self.treasury += tx
                div[self.f_owner[f]] += d - tx
        self.h_money += div
        self.h_div_last = div
        self.f_inventory *= 1 - p.inventory_decay
        self.f_sales = sold
        self.f_lost = lost

        # 10. Insolvenz: Firma kann keinen Lohn mehr zahlen oder ist überschuldet.
        #     Die Bank verwertet die Kasse (ohne Rechtsdurchsetzung nur teilweise),
        #     der Rest der Forderung wird abgeschrieben.
        bankrupt = 0
        writeoff = 0.0
        recovery = 1.0 if p.property_rights else p.recovery_free
        for f in alive:
            assets = self.f_money[f] + self.f_inventory[f] * self.f_price[f]
            if assets < self.f_wage[f] or assets < self.f_debt[f]:
                emp[emp == f] = UNEMPLOYED
                rec = min(self.f_debt[f], max(self.f_money[f], 0.0) * recovery)
                self.f_money[f] -= rec
                writeoff += self.f_debt[f] - rec
                self.f_debt[f] = 0.0
                self.f_rate[f] = 0.0
                self.h_money[self.f_owner[f]] += self.f_money[f]
                self.f_money[f] = 0.0
                self.f_inventory[f] = 0.0
                self.f_sales[f] = 0.0
                self.f_alive[f] = False
                bankrupt += 1

        # 11. Markteintritt (frei bei Wettbewerbsaufsicht, sonst gehemmt)
        alive_now = np.flatnonzero(self.f_alive)
        mean_target = self.f_target[alive_now].mean() if len(alive_now) else 1.0
        entry_prob = 1.0 if p.antitrust else p.entry_prob_free
        capital = p.entry_capital * self.avg_wage / p.init_wage  # real konstantes Startkapital
        for f in np.flatnonzero(~self.f_alive):
            if rng.random() >= entry_prob:
                continue
            # Gründer bringt bis zur Hälfte seines Geldes ein; Rest ggf. per Bankkredit
            rich = np.flatnonzero(self.h_money >= 0.5 * capital)
            if len(rich) == 0:
                break
            inv = rng.choice(rich, p=self.h_money[rich] / self.h_money[rich].sum())
            equity = min(capital, 0.5 * self.h_money[inv])
            if equity < 0.5 * capital and not p.credit:
                continue
            self.h_money[inv] -= equity
            self.f_money[f] = equity
            self.f_debt[f] = 0.0
            self.f_rate[f] = 0.0
            self.f_inventory[f] = 0.0
            self.f_last_rationed[f] = -10**6
            self.f_uid[f] = self.next_uid
            if p.credit and equity < capital:
                self._loan(f, capital - equity)
            if self.f_money[f] < 0.5 * capital:  # Finanzierung gescheitert
                self.h_money[inv] += self.f_money[f] - self.f_debt[f]
                self.f_money[f] = 0.0
                # gewährten Kredit sofort zurückführen (Geld wird wieder vernichtet)
                self.f_debt[f] = 0.0
                continue
            self.f_alive[f] = True
            self.f_uid[f] = self.next_uid
            self.next_uid += 1
            self.f_owner[f] = inv
            self.f_price[f] = self.avg_wage / p.productivity * (1 + p.init_markup)
            self.f_wage[f] = self.avg_wage
            self.f_target[f] = mean_target
            self.f_prod_last[f] = mean_target
            self.f_inventory[f] = 0.0
            self.f_sales[f] = 0.0
            self.f_stockout[f] = False
            self.f_unfilled[f] = True
            self.f_lost[f] = 0.0
            self.f_markup[f] = p.init_markup
            self.f_honest[f] = rng.random() >= p.dishonest_share

        # 11b. Bank: Gewinn, Verluste, Krise/Rettung, Dividenden
        bank_profit = interest_income - writeoff
        self.bank_equity += bank_profit
        rescue, bail_in, crisis, recap = 0.0, 0.0, 0, 0.0
        L = self.loans_total()
        # Private Rekapitalisierung: Eigentümer schießen Kapital nach, wenn es knapp wird
        target_eq = max(2 * p.capital_requirement * L, p.bank_min_equity * self.money_total())
        if self.bank_equity < target_eq:
            own = self.bank_owners
            avail = p.bank_recap_share * np.clip(self.h_money[own], 0, None)
            recap = min(target_eq - self.bank_equity, avail.sum())
            if recap > 0:
                self.h_money[own] -= recap * avail / avail.sum()
                self.bank_equity += recap
        if self.bank_equity < 0:
            crisis = 1
            if p.deposit_insurance:
                # Staat stellt das geforderte Eigenkapital wieder her (soweit Kasse reicht)
                need = p.capital_requirement * L - self.bank_equity
                rescue = min(need, max(self.treasury, 0.0))
                self.treasury -= rescue
                self.bank_equity += rescue
            if self.bank_equity < 0:
                # Verlust trifft die Einleger: anteilige Kürzung der Haushaltsguthaben
                bail_in = -self.bank_equity
                self.h_money -= bail_in * self.h_money / self.h_money.sum()
                self.bank_equity = 0.0
        floor_eq = 1.5 * p.capital_requirement * L
        if bank_profit > 0 and self.bank_equity > floor_eq:
            bd = min(p.bank_dividend_share * bank_profit, self.bank_equity - floor_eq)
            self.bank_equity -= bd
            tx = bd * p.tax_rate
            self.treasury += tx
            self.h_money[self.bank_owners] += (bd - tx) / len(self.bank_owners)
            div[self.bank_owners] += (bd - tx) / len(self.bank_owners)

        # 12. Aggregate und Kennzahlen
        prev_index = self.price_index
        if charged.sum() > EPS:
            self.price_index = revenue.sum() / charged.sum()
        firm_emp = np.bincount(emp[emp >= 0], minlength=F)
        if firm_emp.sum() > 0:
            self.avg_wage = float((self.f_wage * firm_emp).sum() / firm_emp.sum())
        alive_now = np.flatnonzero(self.f_alive)
        self.inflation_last = self.price_index / prev_index - 1
        self.unemployment_last = float((emp == UNEMPLOYED).mean())
        self.records.append({
            "t": t,
            "konsum_pk": consumption.sum() / H,
            "produktion": output,
            "wohlfahrt": float(np.log1p(consumption).mean()),
            "arbeitslosigkeit": float((emp == UNEMPLOYED).mean()),
            "staatsquote": float((emp == PUBLIC).mean()),
            "preisindex": self.price_index,
            "inflation": self.price_index / prev_index - 1,
            "reallohn": self.avg_wage / self.price_index,
            "gini_vermoegen": gini(self.wealth()),
            "gini_einkommen": gini(income + div),
            "gini_konsum": gini(consumption),
            "knappheit": unmet / budget_total if budget_total > 0 else 0.0,
            "hhi": hhi(self.f_sales[alive_now]),
            "firmen": len(alive_now),
            "insolvenzen": bankrupt,
            "betrugsquote": fraud_value / contract_value if contract_value > 0 else 0.0,
            "arbeitslosengeld": self.benefit_last,
            "geldmenge": self.money_total(),
            "kredite": L,
            "kreditquote": L / (12 * self.price_index * output) if output > 0 else np.nan,
            "neukredite": self.c_new,
            "zins": float((self.f_rate * self.f_debt).sum() / L) if L > EPS else np.nan,
            "rationierung": self.c_rationed / self.c_requests if self.c_requests else 0.0,
            "ausfaelle": writeoff / loans_start if loans_start > EPS else 0.0,
            "bank_ek_quote": self.bank_equity / L if L > EPS else np.nan,
            "bankenkrise": crisis,
            "bankenrettung": rescue,
            "bail_in": bail_in,
            "rekapitalisierung": recap,
            "bilanzfehler": self.balance_error(),
        })
        self.t += 1

    def run(self) -> pd.DataFrame:
        for _ in range(self.p.periods):
            self.step()
        return pd.DataFrame(self.records)

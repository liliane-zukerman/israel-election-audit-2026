#!/usr/bin/env python3
"""Counting-centre test: are the anomalies, if any, concentrated in one regional committee (column סמל ועדה)?

Protocols are typed into the system at the regional committees. Fabrication or manipulation at one centre would
show up as digit irregularities or a jump in invalid ballots in that committee's boxes only. For each committee:
  1. last-digit test of the voters count and of every large list's count (boxes with 50+ votes for the list),
     against a binomial simulation of the same boxes (same method as sim_null.py), pooled over the series;
  2. invalid-ballot rate, compared with the same committee in 2022.
usage: python3 committee_check.py [dados/expb.csv dados/expb_2022.csv]"""
import os
import sys

import numpy as np
import pandas as pd

D = "data" if os.path.isdir("data") else "dados"
F26 = sys.argv[1] if len(sys.argv) > 1 else f"{D}/expb.csv"
F22 = sys.argv[2] if len(sys.argv) > 2 else f"{D}/expb_2022.csv"
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
NSIM = 2000
rng = np.random.default_rng(5)


def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    return b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()


def chi(vals):
    obs = np.bincount(vals % 10, minlength=10)
    e = len(vals) / 10
    return ((obs - e) ** 2 / e).sum()


def committee_digits(d):
    big = [p for p in d.columns if p not in META and d[p].sum() > 200000]
    rows = []
    for c, g in d.groupby('סמל ועדה'):
        series = [(g['מצביעים'].values, g['בזב'].values, (g['מצביעים'] / g['בזב']).values)]
        for p in big:
            mask = g[p].values >= 50
            if mask.sum() >= 30:
                series.append((g[p].values[mask], g['כשרים'].values[mask], (g[p] / g['כשרים'].clip(lower=1)).values[mask]))
        obs = sum(chi(s[0]) for s in series)
        sims = np.array([sum(chi(rng.binomial(n, pr)) for _, n, pr in series) for _ in range(NSIM)])
        rows.append(dict(committee=c, boxes=len(g), series=len(series), p_sim=round((1 + (sims >= obs).sum()) / (NSIM + 1), 3),
                         invalid_pct=round(g['פסולים'].sum() / g['מצביעים'].sum() * 100, 2)))
    return pd.DataFrame(rows).set_index("committee")


b, a = load(F26), load(F22)
r26, r22 = committee_digits(b), committee_digits(a)
r = r26.join(r22[["p_sim", "invalid_pct"]], rsuffix="_2022")
r["invalid_change"] = (r.invalid_pct - r.invalid_pct_2022).round(2)
print("regional committees: last-digit test (simulated baseline) and invalid-ballot rate, 2026 vs 2022")
print(r.sort_values("p_sim").to_string())
print(f"\ncommittees with p<0.05: 2026 {int((r.p_sim < .05).sum())} of {len(r)} | 2022 {int((r.p_sim_2022 < .05).sum())} of {len(r)} (chance: about 1)")
print(f"invalid ballots: 2026 {b['פסולים'].sum() / b['מצביעים'].sum() * 100:.2f}% | 2022 {a['פסולים'].sum() / a['מצביעים'].sum() * 100:.2f}% | "
      f"committee change from {r.invalid_change.min():+.2f} to {r.invalid_change.max():+.2f} pp")

#!/usr/bin/env python3
"""Arab-majority ballot boxes (Arab parties >= 70% of valid votes): turnout distribution, turnout x share
association within locality, and last digit of Arab-party counts, for 2022 and 2026."""
import numpy as np
import pandas as pd
from scipy import stats

FILES = {"2022": ("dados/expb_2022.csv", ["ום", "עם", "ד"]), "2026": ("dados/expb.csv", ["ודם", "עם"])}
for year, (f, parts) in FILES.items():
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()  # drop double-envelope boxes
    b["t"] = b['מצביעים'] / b['בזב']
    b["arab"] = b[parts].sum(axis=1) / b['כשרים'].clip(lower=1)
    ar = b[b.arab >= .7]
    within = np.corrcoef(ar.t - ar.groupby('סמל ישוב').t.transform('mean'),
                         ar.arab - ar.groupby('סמל ישוב').arab.transform('mean'))[0, 1]
    print(f"\n{year}: {len(ar)} boxes | voters {ar['מצביעים'].sum()} | median turnout {ar.t.median():.3f} | "
          f"max {ar.t.max():.3f} | boxes >= 90%: {(ar.t >= .9).sum()}")
    print(f"  turnout quantiles: {ar.t.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()}")
    print(f"  corr turnout x Arab share within locality: {within:+.3f}")
    for p in parts + ['מצביעים']:
        v = ar[p].to_numpy()
        v = v[v >= 50]
        if len(v) < 300:
            continue
        obs = np.bincount(v % 10, minlength=10)
        print(f"  last digit {p}: n={len(v)} p={stats.chisquare(obs).pvalue:.3f}")

# Voter register: did the electorate of Arab-majority localities grow abnormally between 2022 and 2026?
# A surge manufactured through the register would show up here; demographic growth is a few percent a year.
loc = {}
for year, (f, parts) in FILES.items():
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)]
    g = b.groupby('סמל ישוב').agg(elig=('בזב', 'sum'), voters=('מצביעים', 'sum'), valid=('כשרים', 'sum'))
    g["arab"] = b.groupby('סמל ישוב')[parts].sum().sum(axis=1) / g.valid.clip(lower=1)
    loc[year] = g
m = loc["2022"].join(loc["2026"], lsuffix="_22", rsuffix="_26", how="inner")
m["group"] = np.where((m.arab_22 >= .7) & (m.arab_26 >= .7), "Arab-majority",
                      np.where((m.arab_22 < .3) & (m.arab_26 < .3), "Jewish-majority", "mixed"))
t = m.groupby("group").agg(localities=("group", "size"), elig_22=("elig_22", "sum"), elig_26=("elig_26", "sum"),
                           voters_22=("voters_22", "sum"), voters_26=("voters_26", "sum"))
t["register_growth_%"] = (t.elig_26 / t.elig_22 - 1) * 100
t["turnout_22_%"] = t.voters_22 / t.elig_22 * 100
t["turnout_26_%"] = t.voters_26 / t.elig_26 * 100
print("\nvoter register and turnout by locality group (regular boxes, localities present in both years):")
print(t.round(1).to_string())

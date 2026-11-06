#!/usr/bin/env python3
"""Haredi-majority ballot boxes (Haredi lists >= 70% of valid votes): turnout distribution, turnout x share
association within locality, last digit of Haredi-party counts, and comparison of 2022 and 2026."""
import numpy as np
import pandas as pd
from scipy import stats

# Haredi lists: Shas and UTJ in both years; small Haredi lists that ran in 2026 (Tzeva Shachor, HaTzibur HaCharedi)
FILES = {"2022": ("dados/expb_2022.csv", ["שס", "ג"]), "2026": ("dados/expb.csv", ["שס", "ג", "נר", "זך"])}
for year, (f, parts) in FILES.items():
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()  # drop double-envelope boxes
    b["t"] = b['מצביעים'] / b['בזב']
    b["har"] = b[parts].sum(axis=1) / b['כשרים'].clip(lower=1)
    h = b[b.har >= .7]
    within = np.corrcoef(h.t - h.groupby('סמל ישוב').t.transform('mean'),
                         h.har - h.groupby('סמל ישוב').har.transform('mean'))[0, 1]
    print(f"\n{year}: {len(h)} boxes | voters {h['מצביעים'].sum()} | median turnout {h.t.median():.3f} | "
          f"max {h.t.max():.3f} | boxes >= 90%: {(h.t >= .9).sum()} | boxes <= 30%: {(h.t <= .3).sum()}")
    print(f"  turnout quantiles: {h.t.quantile([.05, .25, .5, .75, .95]).round(3).to_dict()}")
    print(f"  corr turnout x Haredi share within locality: {within:+.3f}")
    print(f"  UTJ share of Haredi vote: {h['ג'].sum() / h[parts].sum().sum():.3f}")
    for p in ["שס", "ג", 'מצביעים']:
        v = h[p].to_numpy()
        v = v[v >= 50]
        if len(v) < 300:
            continue
        obs = np.bincount(v % 10, minlength=10)
        print(f"  last digit {p}: n={len(v)} p={stats.chisquare(obs).pvalue:.3f}")
    top = h.groupby('שם ישוב').agg(boxes=('t', 'size'), voters=('מצביעים', 'sum'), turnout=('t', 'median'))
    print("  largest localities:", top.sort_values('voters', ascending=False).head(6).round(3).to_dict('index'))
    low = h[h.t <= .3]
    if len(low):
        print("  very low turnout boxes by locality:", low['שם ישוב'].value_counts().head(6).to_dict())

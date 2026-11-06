#!/usr/bin/env python3
"""Extra tests: (1) invalid votes 2022 vs 2026 in matched boxes, (2) invalid rate vs each party's share within
the same locality (a sign of ballots of one party being disqualified), (3) integer-percentage spikes in party
shares (Kobak test applied to vote shares), (4) boxes whose turnout changed much more than their locality."""
import os
import numpy as np
import pandas as pd

DATA = "data" if os.path.isdir("data") else "dados"
rng = np.random.default_rng(21)
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']


def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    P = [c for c in b.columns if c not in META]
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()
    b["key"] = b['סמל ישוב'].astype(str) + "/" + b['קלפי'].astype(float).astype(str)
    b["inv"] = b['פסולים'] / b['מצביעים'].clip(lower=1)
    b["t"] = b['מצביעים'] / b['בזב']
    return b, P


a, Pa = load(f"{DATA}/expb_2022.csv")
b, Pb = load(f"{DATA}/expb.csv")
top = b[Pb].sum().sort_values(ascending=False).index[:11].tolist()

print("== 1. Invalid votes ==")
for y, d in (("2022", a), ("2026", b)):
    print(f"{y}: rate {100*d['פסולים'].sum()/d['מצביעים'].sum():.2f}% | boxes with >= 3%: {(d.inv >= .03).sum()} | max {100*d.inv.max():.1f}%")
m = a.merge(b, on="key", suffixes=("_22", "_26"))
m = m[(m['מצביעים_22'] >= 100) & (m['מצביעים_26'] >= 100)]
# expected spread of the change under binomial noise
se = np.sqrt(m.inv_22 * (1 - m.inv_22) / m['מצביעים_22'] + m.inv_26.clip(lower=.002) * (1 - m.inv_26) / m['מצביעים_26'])
m["z_inv"] = (m.inv_26 - m.inv_22 - (m.inv_26 - m.inv_22).median()) / se
print(f"matched boxes {len(m)} | invalid rate jumped by > 4 sd: {(m.z_inv > 4).sum()} | dropped by > 4 sd: {(m.z_inv < -4).sum()}")
print(m.sort_values("z_inv", ascending=False)[["שם ישוב_26", "קלפי_26", "מצביעים_26", "inv_22", "inv_26", "z_inv"]].head(8).round(3).to_string(index=False))

print("\n== 2. Invalid rate vs party share within the same locality (2026) ==")
d = b[b['מצביעים'] >= 100].copy()
inv_dev = d.inv - d.groupby('סמל ישוב').inv.transform('mean')
rows = []
for p in top:
    sh = d[p] / d['כשרים'].clip(lower=1)
    dev = sh - sh.groupby(d['סמל ישוב']).transform('mean')
    obs = np.corrcoef(inv_dev, dev)[0, 1]
    # permutation of invalid rates within locality
    perm = []
    g = d.groupby('סמל ישוב').indices
    for _ in range(200):
        x = inv_dev.to_numpy().copy()
        for idx in g.values():
            x[idx] = x[rng.permutation(idx)]
        perm.append(np.corrcoef(x, dev)[0, 1])
    perm = np.array(perm)
    rows.append((p, round(obs, 3), round(perm.std(), 3), round(obs / perm.std(), 1)))
print(pd.DataFrame(rows, columns=["party", "corr", "sd_null", "z"]).to_string(index=False))

print("\n== 3. Integer percentages in party shares (2026) ==")
for p in top[:6]:
    sh = 100 * b[p] / b['כשרים'].clip(lower=1)
    sel = (b['כשרים'] >= 100) & (sh >= 5) & (sh <= 95)
    obs = int((np.abs(sh[sel] - sh[sel].round()) < .05).sum())
    sims = []
    n, s = b.loc[sel, 'כשרים'].to_numpy(), (sh[sel] / 100).to_numpy()
    for _ in range(200):
        x = 100 * rng.binomial(n, s) / n
        sims.append(int((np.abs(x - np.round(x)) < .05).sum()))
    sims = np.array(sims)
    print(f"  {p}: {obs} boxes on an integer % | simulated {sims.mean():.0f} [p95 {np.percentile(sims, 95):.0f}]")

print("\n== 4. Turnout change: box vs its locality (matched boxes) ==")
m["dt"] = m.t_26 - m.t_22
m["dt_loc"] = m.groupby('סמל ישוב_26').dt.transform('median')
se_t = np.sqrt(m.t_22 * (1 - m.t_22) / m['בזב_22'] + m.t_26 * (1 - m.t_26) / m['בזב_26'])
spread = (m.dt - m.dt_loc).std()
m["z_t"] = (m.dt - m.dt_loc) / np.sqrt(se_t ** 2 + (spread * .5) ** 2)
print(f"boxes whose turnout change differs from their locality by > 4 sd: up {(m.z_t > 4).sum()}, down {(m.z_t < -4).sum()}")
print(m.sort_values("z_t", ascending=False)[["שם ישוב_26", "קלפי_26", "t_22", "t_26", "dt_loc", "z_t"]].head(8).round(3).to_string(index=False))

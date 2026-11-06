#!/usr/bin/env python3
"""Compare 2026 with 2022 by ballot box and by locality, using voting blocs (parties changed between elections)."""
import numpy as np
import pandas as pd

def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()
    b["key"] = b['סמל ישוב'].astype(str) + "/" + b['קלפי'].astype(float).astype(str)
    return b

a, b = load("dados/expb_2022.csv"), load("dados/expb.csv")
BLOCS = {
    "arab": (["ום", "עם", "ד"], ["ודם", "עם"]),
    "haredi": (["שס", "ג"], ["שס", "ג"]),
    "netanyahu_coalition": (["מחל", "ט", "שס", "ג"], ["מחל", "ב", "ט", "שס", "ג"]),
    "likud": (["מחל"], ["מחל"]),
}
for name, (p22, p26) in BLOCS.items():
    a[name] = a[p22].sum(axis=1) / a['כשרים'].clip(lower=1)
    b[name] = b[p26].sum(axis=1) / b['כשרים'].clip(lower=1)
a["turnout"], b["turnout"] = a['מצביעים'] / a['בזב'], b['מצביעים'] / b['בזב']

# Box-level match (same locality code and box number)
m = a.merge(b, on="key", suffixes=("_22", "_26"))
print(f"boxes 2022 {len(a)} | 2026 {len(b)} | matched by locality+box number {len(m)}")
for k in ["turnout"] + list(BLOCS):
    print(f"  corr {k:20s} box 2022 x 2026: {np.corrcoef(m[k+'_22'], m[k+'_26'])[0,1]:.3f}")

# Locality level
def city(d):
    g = d.groupby('סמל ישוב')
    out = g[['בזב', 'מצביעים', 'כשרים']].sum()
    for name in BLOCS:
        out[name] = (d[name] * d['כשרים']).groupby(d['סמל ישוב']).sum() / out['כשרים']
    out["turnout"] = out['מצביעים'] / out['בזב']
    out["nome"] = g['שם ישוב'].first()
    return out
ca, cb = city(a), city(b)
c = ca.add_suffix("_22").join(cb.add_suffix("_26"), how="inner")
c = c[c['כשרים_26'] >= 500]
print(f"\nlocalities with 500+ valid votes in both: {len(c)}")
for k in ["turnout"] + list(BLOCS):
    w = c['כשרים_26']
    d = c[k+'_26'] - c[k+'_22']
    print(f"  {k:20s} corr {np.corrcoef(c[k+'_22'], c[k+'_26'])[0,1]:.3f} | mean change {100*np.average(d, weights=w):+.1f} pp")

# Boxes flagged in 2026: were they the same in 2022?
print("\n2026 outlier boxes vs same box in 2022:")
look = [("רחובות", 142.1, "עם"), ("כפר סבא", 6.1, "עם"), ("תל אביב", 226.2, "ג"), ("נתניה", 147.0, "ג"), ("באר שבע", 217.0, "שס")]
for city_name, box, p in look:
    r26 = b[(b['שם ישוב'].str.contains(city_name.split()[0])) & (b['קלפי'] == box)]
    if r26.empty:
        continue
    key = r26.key.iloc[0]
    r22 = a[a.key == key]
    s26 = 100 * r26[p].iloc[0] / r26['כשרים'].iloc[0]
    s22 = 100 * r22[p].iloc[0] / r22['כשרים'].iloc[0] if (not r22.empty and p in a) else float('nan')
    print(f"  {city_name} box {box} party {p}: 2026 {s26:.1f}% | 2022 {s22:.1f}% (valid {int(r22['כשרים'].iloc[0]) if not r22.empty else '-'})")

# Largest locality-level changes in turnout and in Arab bloc
c["d_turnout"] = 100 * (c.turnout_26 - c.turnout_22)
print("\nlargest turnout changes (localities):")
print(c.sort_values("d_turnout")[["nome_26", "turnout_22", "turnout_26", "d_turnout", 'כשרים_26']].round(3).iloc[np.r_[0:6, -6:0]].to_string())
c.to_pickle("dados/cidades.pkl"); m.to_pickle("dados/urnas_pareadas.pkl")

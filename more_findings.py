#!/usr/bin/env python3
"""Additional findings: seats (preliminary), effect of the Arab turnout surge on seats, where the extra voters came
from, war-affected border communities, wasted votes below the threshold, invalid votes."""
import os
import numpy as np
import pandas as pd

DATA = "data" if os.path.isdir("data") else "dados"
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
NAME = {"מחל": "Likud", "דרך": "Yashar!", "רק": "Beyachad", "ודם": "Joint List", "אמת": "Democrats",
        "ל": "Yisrael Beiteinu", "עם": "Ra'am", "ב": "Otzma Yehudit", "ג": "UTJ", "שס": "Shas", "ט": "Rel. Zionism",
        "די": "Reservists", "ך": "Amcha Yisrael"}
COAL = ["מחל", "ב", "ט", "שס", "ג"]
ARAB = ["ודם", "עם"]


def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    return b, [c for c in b.columns if c not in META]


def seats(votes, threshold=.0325, n=120):
    """Bader-Ofer (D'Hondt) among lists above the threshold; surplus agreements ignored."""
    valid = votes.sum()
    ok = votes[votes >= threshold * valid]
    s = pd.Series(0, index=ok.index)
    for _ in range(n):
        s[(ok / (s + 1)).idxmax()] += 1
    return s


b26, P26 = load(f"{DATA}/expb.csv")
b22, P22 = load(f"{DATA}/expb_2022.csv")
v = b26[P26].sum()
st = seats(v)
print("== Seats from the preliminary count (no surplus agreements) ==")
print(" | ".join(f"{NAME.get(p, p)} {k}" for p, k in st.sort_values(ascending=False).items()))
coal = st.reindex(COAL).fillna(0).sum()
arab = st.reindex(ARAB).fillna(0).sum()
print(f"Netanyahu coalition parties: {coal:.0f} | Arab parties: {arab:.0f} | other: {120 - coal - arab:.0f}")
below = v[v < .0325 * v.sum()]
print(f"votes for lists below the threshold: {below.sum():,} ({100 * below.sum() / v.sum():.1f}%); "
      f"Reservists short of the threshold by {int(.0325 * v.sum() - v['די']):,} votes")

# Counterfactual: Arab-majority boxes back to their 2022 turnout (same vote shares inside each box group)
cbs = pd.read_excel(f"{DATA}/bycode2021.xlsx").set_index('סמל יישוב')
rel = cbs['דת יישוב']
reg = b26[(b26['סמל ישוב'] != 9999) & (b26['בזב'] > 0)].copy()
reg22 = b22[(b22['סמל ישוב'] != 9999) & (b22['בזב'] > 0)].copy()
for d, parts in ((reg, ARAB), (reg22, ["ום", "עם", "ד"])):
    d["arab_box"] = d[parts].sum(axis=1) / d['כשרים'].clip(lower=1) >= .5
t22 = reg22[reg22.arab_box].groupby('סמל ישוב').apply(lambda d: d['מצביעים'].sum() / d['בזב'].sum(), include_groups=False)
cf = reg[reg.arab_box].copy()
cf["t26"] = cf.groupby('סמל ישוב')['מצביעים'].transform('sum') / cf.groupby('סמל ישוב')['בזב'].transform('sum')
cf["t22"] = cf['סמל ישוב'].map(t22).fillna(reg22[reg22.arab_box]['מצביעים'].sum() / reg22[reg22.arab_box]['בזב'].sum())
factor = (cf.t22 / cf.t26).clip(upper=1)
removed = cf[P26].mul(1 - factor, axis=0).sum()
v_cf = v - removed
st_cf = seats(v_cf)
print(f"\n== If Arab-majority boxes had their 2022 turnout ==\nvotes removed: {int(removed.sum()):,} "
      f"(of which Arab parties {int(removed[ARAB].sum()):,})")
print(" | ".join(f"{NAME.get(p, p)} {k}" for p, k in st_cf.sort_values(ascending=False).items()))
print(f"Netanyahu coalition parties: {st_cf.reindex(COAL).fillna(0).sum():.0f} | Arab parties: {st_cf.reindex(ARAB).fillna(0).sum():.0f}")

# Where the extra voters came from
reg["group"] = np.where(reg.arab_box, "Arab-majority boxes", "other boxes")
reg22["group"] = np.where(reg22.arab_box, "Arab-majority boxes", "other boxes")
g26, g22 = reg.groupby("group")['מצביעים'].sum(), reg22.groupby("group")['מצביעים'].sum()
e26 = b26[b26['סמל ישוב'] == 9999]['מצביעים'].sum(); e22 = b22[b22['סמל ישוב'] == 9999]['מצביעים'].sum()
tot = (g26 - g22).sum() + (e26 - e22)
print(f"\n== Extra voters 2022 -> 2026: {tot:,} ==")
for k in g26.index:
    print(f"  {k}: {g26[k] - g22[k]:+,} ({100 * (g26[k] - g22[k]) / tot:.0f}%)")
print(f"  double envelopes: {e26 - e22:+,}")
print(f"  eligible voters: {b22['בזב'].sum():,} -> {b26['בזב'].sum():,}")

# War-affected border communities (regional councils + towns)
council = cbs['שם מעמד מונציפאלי']
GAZA = {"מועצה אזורית אשכול", "מועצה אזורית שער הנגב", "מועצה אזורית שדות נגב", "מועצה אזורית חוף אשקלון"}
NORTH = {"מועצה אזורית מטה אשר", "מועצה אזורית מעלה יוסף", "מועצה אזורית מרום הגליל", "מועצה אזורית הגליל העליון",
         "מועצה אזורית מבואות החרמון"}
TOWNS_G = {"שדרות"}; TOWNS_N = {"קרית שמונה", "מטולה", "שלומי"}
def area(d):
    c = d['סמל ישוב'].map(council)
    a = pd.Series("rest", index=d.index)
    a[c.isin(GAZA) | d['שם ישוב'].isin(TOWNS_G)] = "Gaza border"
    a[c.isin(NORTH) | d['שם ישוב'].isin(TOWNS_N)] = "Northern border"
    return a
print("\n== War-affected border communities (share of valid votes) ==")
for yr, d, coal_p, P in (("2022", reg22, ["מחל", "ט", "שס", "ג"], P22), ("2026", reg, COAL, P26)):
    d["area"] = area(d)
    for k, g in d.groupby("area"):
        vv = g['כשרים'].sum()
        print(f"  {yr} {k:16s} boxes {len(g):5d} turnout {100*g['מצביעים'].sum()/g['בזב'].sum():.1f}% | "
              f"coalition {100*g[coal_p].sum().sum()/vv:.1f}% | Likud {100*g['מחל'].sum()/vv:.1f}%"
              + (f" | Yashar {100*g['דרך'].sum()/vv:.1f}% | Beyachad {100*g['רק'].sum()/vv:.1f}% | Otzma {100*g['ב'].sum()/vv:.1f}%" if yr == "2026" else ""))

print(f"\n== Invalid votes ==\n2022 {100*b22['פסולים'].sum()/b22['מצביעים'].sum():.2f}% | 2026 {100*b26['פסולים'].sum()/b26['מצביעים'].sum():.2f}%")

#!/usr/bin/env python3
"""Second detector for transcription errors, independent of polling-site size.

A box is flagged when a fringe list (under 0.5% nationally) holds 10% or more of its valid votes, that list has
under 1% in the rest of the locality, and some major list that is strong in the locality (15%+ in its other boxes,
or in the same locality in 2022) is almost absent in the box (2% or less). The votes of the major list were most
likely recorded under the fringe list's letters. Also reports where the two lists sit in the alphabetical order of
the protocol: adjacent rows are the classic slip."""
import os
import numpy as np
import pandas as pd

DATA = "data" if os.path.isdir("data") else "dados"
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
BLOC22 = {"ודם": ["ום", "ד"], "עם": ["עם"], "מחל": ["מחל"], "ל": ["ל"], "שס": ["שס"], "ג": ["ג"], "ט": ["ט"],
          "אמת": ["אמת", "מרצ"], "דרך": ["כן", "פה"], "רק": ["פה"], "ב": ["ט"]}


def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    return b[b['סמל ישוב'] != 9999].copy(), [c for c in b.columns if c not in META]


a, PA = load(f"{DATA}/expb_2022.csv")
b, P = load(f"{DATA}/expb.csv")
pos = {p: i for i, p in enumerate(P)}
tot = b[P].sum()
fringe = [p for p in P if 0 < tot[p] < 0.005 * tot.sum()]
major = [p for p in P if tot[p] >= 0.03 * tot.sum()]
share = b[P].div(b['כשרים'].clip(lower=1), axis=0)
loc_tot = b.groupby('סמל ישוב')[P + ['כשרים']].transform('sum')
rest = (loc_tot[P].sub(b[P])).div((loc_tot['כשרים'] - b['כשרים']).clip(lower=1), axis=0)
has_rest = (loc_tot['כשרים'] - b['כשרים']) > 0
loc22 = a.groupby('סמל ישוב')[PA + ['כשרים']].sum()
rows = []
for i, r in b.iterrows():
    for f in fringe:
        if share.at[i, f] < .10 or (has_rest[i] and rest.at[i, f] > .01):
            continue
        for m in major:
            strong_here = has_rest[i] and rest.at[i, m] >= .15
            strong_2022 = False
            if r['סמל ישוב'] in loc22.index and m in BLOC22:
                l = loc22.loc[r['סמל ישוב']]
                strong_2022 = l[BLOC22[m]].sum() / max(l['כשרים'], 1) >= .15
            if share.at[i, m] <= .02 and (strong_here or strong_2022):
                rows.append((r['שם ישוב'], r['קלפי'], int(r['כשרים']), m, round(100 * share.at[i, m], 1),
                             round(100 * rest.at[i, m], 1) if has_rest[i] else None, f, int(r[f]),
                             round(100 * share.at[i, f], 1), pos[f] - pos[m]))
out = pd.DataFrame(rows, columns=["locality", "box", "valid", "missing_list", "box_%", "rest_of_locality_%",
                                  "appears_as", "votes", "box_%_fringe", "rows_apart_in_protocol"])
pd.set_option("display.width", 220)
print(f"boxes flagged: {len(out)}")
print(out.sort_values("votes", ascending=False).to_string(index=False))
out.to_csv(f"{DATA}/fringe_check.csv", index=False)

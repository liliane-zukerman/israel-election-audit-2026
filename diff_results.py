#!/usr/bin/env python3
"""Compare two downloads of the results-by-box file (e.g. preliminary vs final): boxes whose numbers changed.
Usage: python3 diff_results.py old_expb.csv new_expb.csv"""
import sys
import pandas as pd

META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
old, new = [pd.read_csv(f, encoding="utf-8-sig") for f in sys.argv[1:3]]
for d in (old, new):
    d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")], inplace=True)
    d["key"] = d['סמל ישוב'].astype(str) + "/" + d['קלפי'].astype(float).astype(str)
P = [c for c in new.columns if c not in META + ["key"]]
m = old.merge(new, on="key", suffixes=("_old", "_new"))
print(f"boxes old {len(old)} | new {len(new)} | matched {len(m)} | added {len(set(new.key) - set(old.key))} | removed {len(set(old.key) - set(new.key))}")
changed = []
for _, r in m.iterrows():
    diffs = {p: (int(r[p + "_old"]), int(r[p + "_new"])) for p in P + ['מצביעים', 'פסולים', 'כשרים'] if r[p + "_old"] != r[p + "_new"]}
    if diffs:
        changed.append((r['שם ישוב_new'], r['קלפי_new'], diffs))
print(f"boxes with any change: {len(changed)}")
for city, box, diffs in changed:
    print(f"  {city} box {box}: " + ", ".join(f"{p} {o}->{n}" for p, (o, n) in diffs.items()))

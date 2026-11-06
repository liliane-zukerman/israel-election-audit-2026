#!/usr/bin/env python3
"""Pattern-specific detector for protocol transcription slips between two lists.

The committee's corrections of 2 November 2026 showed that the common slip is a major list typed under the letters
of a micro-list on a neighbouring row of the protocol (HaKahal "רץ" for Beyachad "רק", Gan Eden "ה" for the Joint
List "ודם", and so on). This script looks only for that: boxes where the micro-list has at least MIN_VOTES votes and
the major list has at most 1, while the major list holds at least REST_SHARE of the valid votes in the other boxes of
the same polling site (column ריכוז) and the micro-list is absent there. Lower thresholds than fringe_check.py, so it
catches smaller slips; the pairs are fixed, so it does not catch new ones.

usage: python3 slip_check.py [dados/expb.csv]"""
import os
import sys

import pandas as pd

PAIRS = [("רץ", "רק", "HaKahal <- Beyachad"), ("ה", "ודם", "Gan Eden <- Joint List"), ("נף", "עם", "נף <- Ra'am"),
         ("נקי", "עם", "נקי <- Ra'am"), ("בד", "ב", "בד <- Otzma Yehudit"), ("ד", "די", "ד <- Reservists"),
         ("ני", "ג", "ני <- UTJ"), ("ני", "שס", "ני <- Shas")]
MIN_VOTES, REST_SHARE = 15, 0.03

f = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data" if os.path.isdir("data") else "dados", "expb.csv")
b = pd.read_csv(f, encoding="utf-8-sig")
b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
b = b[b['סמל ישוב'] != 9999].copy()
b["site"] = b['סמל ישוב'].astype(str) + "/" + b['ריכוז'].astype(str)
g = b.groupby("site")
tot_site = g['כשרים'].transform("sum")
n_site = g["site"].transform("size")
total = 0
for micro, major, name in PAIRS:
    if micro not in b.columns or major not in b.columns:
        continue
    rest_valid = (tot_site - b['כשרים']).clip(lower=1)
    rest_major = (g[major].transform("sum") - b[major]) / rest_valid
    rest_micro = (g[micro].transform("sum") - b[micro]) / rest_valid
    sel = b[(b[micro] >= MIN_VOTES) & (b[major] <= 1) & (n_site >= 2) & (rest_major >= REST_SHARE) & (rest_micro <= 0.01)]
    if len(sel):
        print(f"\n{name}: {len(sel)} boxes, {int(sel[micro].sum())} votes")
        for i, r in sel.iterrows():
            print(f"  {r['שם ישוב']} {r['קלפי']}: {micro}={int(r[micro])} {major}={int(r[major])} | rest of site: {major} "
                  f"{rest_major[i]*100:.1f}%, {micro} {rest_micro[i]*100:.2f}% | valid {int(r['כשרים'])}")
        total += int(sel[micro].sum())
print(f"\ntotal votes in flagged boxes: {total}")

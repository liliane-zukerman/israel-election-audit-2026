#!/usr/bin/env python3
"""Forensic tests by religious / community group, 2022 and 2026.

Groups come from the CBS list of localities (bycode2021.xlsx: 'דת יישוב' 1 = Jewish, 2 = Arab, 3 = Bedouin tribes,
4 = mixed city; 'צורת יישוב שוטפת' 330 = kibbutz), plus named localities for Druze, Christian-majority and Bedouin
towns (the CBS file does not split Arab localities by religion), plus the ballot box's own vote for groups that live
inside larger localities (Haredi, religious-Zionist, Arab and Jewish boxes of mixed cities).

For each group: boxes, voters, turnout, bloc shares, the turnout x winner-share test (Klimek), the within-locality
association between turnout and the group's leading bloc, and a last-digit test of the leading party against a
binomial simulation null.
"""
import os
import numpy as np
import pandas as pd

DATA = "data" if os.path.isdir("data") else "dados"
rng = np.random.default_rng(11)
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']

# Locality codes (CBS / Central Elections Committee share the same codes)
DRUZE = {494, 534, 502, 485, 480, 496, 536, 1296, 1295, 525, 546, 4201, 4203, 4001, 4502}
CHRISTIAN = {535, 518, 487, 530}                        # Fassuta, Mi'ilya, Jish, Eilabun
MIXED_ARAB = {473, 543, 507, 481, 8800}                 # Abu Snan, Rame, Kafr Yasif, Maghar, Shefa-Amr
NORTH_BEDOUIN = {652, 998, 944, 975, 1321, 978, 913}    # Ibtin, Bir al-Maksur, Basmat Tab'un, Zarzir, Khawaled, Ka'abiyye, Shibli

BLOCS = {"2022": {"arab": ["ום", "עם", "ד"], "haredi": ["שס", "ג"], "rel_zionist": ["ט"],
                  "coalition": ["מחל", "ט", "שס", "ג"], "likud": ["מחל"]},
         "2026": {"arab": ["ודם", "עם"], "haredi": ["שס", "ג", "נר", "זך"], "rel_zionist": ["ט", "ב"],
                  "coalition": ["מחל", "ב", "ט", "שס", "ג"], "likud": ["מחל"]}}

cbs = pd.read_excel(f"{DATA}/bycode2021.xlsx")
cbs = cbs.set_index('סמל יישוב')
rel = cbs['דת יישוב']
form = cbs['צורת יישוב שוטפת']
district = cbs['שם מחוז']
negev_bedouin = set(cbs.index[(rel == 3) | ((rel == 2) & (district == 'הדרום'))])


def classify(b, year):
    B = BLOCS[year]
    v = b['כשרים'].clip(lower=1)
    share = {k: b[p].sum(axis=1) / v for k, p in B.items()}
    code = b['סמל ישוב']
    r = code.map(rel)
    g = pd.Series("other_jewish", index=b.index)
    g[(r == 2)] = "muslim_arab_towns"
    g[code.isin(MIXED_ARAB)] = "mixed_religion_arab_towns"
    g[code.isin(DRUZE)] = "druze"
    g[code.isin(CHRISTIAN)] = "christian_majority_villages"
    g[code.isin(NORTH_BEDOUIN)] = "bedouin_north"
    g[code.isin(negev_bedouin)] = "bedouin_negev"
    mixed = r == 4
    g[mixed & (share["arab"] >= .5)] = "mixed_cities_arab_boxes"
    g[mixed & (share["arab"] < .5)] = "mixed_cities_jewish_boxes"
    jewish = (r == 1) | (mixed & (share["arab"] < .5)) | r.isna()
    g[jewish & (code.map(form) == 330)] = "kibbutzim"
    g[jewish & (share["rel_zionist"] >= .4)] = "religious_zionist_boxes"
    g[jewish & (share["haredi"] >= .7)] = "haredi_boxes"
    return g, share


def last_digit_p(x, n, sims=300):
    """Last digit of counts >= 50 vs Binomial(n, observed share) redraws (parametric bootstrap)."""
    keep = x >= 50
    if keep.sum() < 150:
        return None, int(keep.sum())
    share = x / np.clip(n, 1, None)
    def chi(counts, exp):
        e = exp * counts.sum()
        return (((counts - e) ** 2) / e).sum()
    sim_h = []
    for _ in range(sims):
        s = rng.binomial(n, share)
        sim_h.append(np.bincount(s[s >= 50] % 10, minlength=10))
    sim_h = np.array(sim_h)
    exp = sim_h.sum(axis=0) / sim_h.sum()
    obs = np.bincount(x[keep] % 10, minlength=10)
    c = chi(obs, exp)
    return (1 + sum(chi(h, exp) >= c for h in sim_h)) / (sims + 1), int(keep.sum())


rows = []
for year, f in (("2022", f"{DATA}/expb_2022.csv"), ("2026", f"{DATA}/expb.csv")):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    P = [c for c in b.columns if c not in META]
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()
    b["t"] = b['מצביעים'] / b['בזב']
    b["group"], share = classify(b, year)
    for k, s in share.items():
        b[k] = s
    for grp, g in b.groupby("group"):
        tot = g[P].sum()
        lead = tot.idxmax()
        ws = g[P].max(axis=1) / g['כשרים'].clip(lower=1)
        lead_share = g[lead] / g['כשרים'].clip(lower=1)
        within = np.nan
        if g['סמל ישוב'].nunique() < len(g):
            dt = g.t - g.groupby('סמל ישוב').t.transform('mean')
            ds = lead_share - lead_share.groupby(g['סמל ישוב']).transform('mean')
            if dt.std() > 0 and ds.std() > 0:
                within = np.corrcoef(dt, ds)[0, 1]
        p, nd = last_digit_p(g[lead].to_numpy(), g['כשרים'].to_numpy())
        top3 = (100 * tot.sort_values(ascending=False).head(3) / tot.sum()).round(1)
        rows.append({"year": year, "group": grp, "boxes": len(g), "voters": int(g['מצביעים'].sum()),
                     "turnout_%": round(100 * g['מצביעים'].sum() / g['בזב'].sum(), 1),
                     "max_turnout_%": round(100 * g.t.max(), 1),
                     "klimek_corner": int(((g.t >= .9) & (ws >= .9)).sum()),
                     "within_corr": round(within, 2) if pd.notna(within) else None,
                     "lead": lead, "last_digit_p": None if p is None else round(p, 3), "digit_n": nd,
                     "top3": ", ".join(f"{k} {v}" for k, v in top3.items()),
                     **{f"{k}_%": round(100 * g[BLOCS[year][k]].sum().sum() / g['כשרים'].sum(), 1)
                        for k in ("arab", "haredi", "coalition")}})

r = pd.DataFrame(rows)
order = ["muslim_arab_towns", "bedouin_negev", "bedouin_north", "druze", "christian_majority_villages",
         "mixed_religion_arab_towns", "mixed_cities_arab_boxes", "mixed_cities_jewish_boxes", "haredi_boxes",
         "religious_zionist_boxes", "kibbutzim", "other_jewish"]
r["o"] = r.group.map({g: i for i, g in enumerate(order)})
r = r.sort_values(["o", "year"]).drop(columns="o")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(r[["group", "year", "boxes", "voters", "turnout_%", "max_turnout_%", "klimek_corner", "within_corr",
         "last_digit_p", "digit_n", "arab_%", "haredi_%", "coalition_%"]].to_string(index=False))
print("\nleading parties by group (2026):")
print(r[r.year == "2026"][["group", "top3"]].to_string(index=False))
r.to_csv(f"{DATA}/religious_groups.csv", index=False)

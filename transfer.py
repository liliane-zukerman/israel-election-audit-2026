#!/usr/bin/env python3
"""Vote transfer 2022 -> 2026: constrained R x C ecological model.

Each unit (locality, or ballot box matched by number) is an equation: the 2026 vote shares of the electorate are a
mix of the 2022 shares, with an unknown transfer matrix B (rows = 2022 choice, columns = 2026 choice, each row sums
to 1, all entries between 0 and 1). B is the matrix that best explains all units at once (weighted least squares).
This is an estimate of average behaviour (ecological inference), not a count of individual voters.
"""
import os
import sys
import cvxpy as cp
import numpy as np
import pandas as pd

DATA = "data" if os.path.isdir("data") else "dados"
R = {"Likud": ["מחל"], "Yesh Atid": ["פה"], "Rel. Zionism+Otzma": ["ט"], "National Unity": ["כן"], "Shas": ["שס"],
     "UTJ": ["ג"], "Yisrael Beiteinu": ["ל"], "Labor+Meretz": ["אמת", "מרצ"], "Arab parties": ["ום", "עם", "ד"],
     "Other 2022": None, "Did not vote 2022": "abst"}
C = {"Likud": ["מחל"], "Yashar!": ["דרך"], "Beyachad": ["רק"], "Arab parties": ["ודם", "עם"], "Democrats": ["אמת"],
     "Yisrael Beiteinu": ["ל"], "Otzma Yehudit": ["ב"], "UTJ": ["ג"], "Shas": ["שס"], "Rel. Zionism": ["ט"],
     "Reservists": ["די"], "Other 2026": None, "Did not vote 2026": "abst"}
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']


def load(f):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    P = [c for c in b.columns if c not in META]
    return b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy(), P


def shares(d, P, spec):
    """Shares of the electorate for each category (votes for parties, other valid+invalid, abstention)."""
    out = {}
    used = []
    for k, cols in spec.items():
        if cols is None or cols == "abst":
            continue
        out[k] = d[cols].sum(axis=1)
        used += cols
    other = [k for k, v in spec.items() if v is None][0]
    out[other] = d[P].sum(axis=1) - sum(out.values()) + d['פסולים']
    out[[k for k, v in spec.items() if v == "abst"][0]] = d['בזב'] - d['מצביעים']
    m = pd.DataFrame(out)[list(spec)]
    return m.div(d['בזב'], axis=0)


level = sys.argv[1] if len(sys.argv) > 1 else "locality"
a, Pa = load(f"{DATA}/expb_2022.csv")
b, Pb = load(f"{DATA}/expb.csv")
if level == "locality":
    num = ['בזב', 'מצביעים', 'פסולים', 'כשרים']
    a = a.groupby('סמל ישוב')[num + Pa].sum().reset_index()
    b = b.groupby('סמל ישוב')[num + Pb].sum().reset_index()
    key = 'סמל ישוב'
else:
    for d in (a, b):
        d["key"] = d['סמל ישוב'].astype(str) + "/" + d['קלפי'].astype(float).astype(str)
    key = "key"
X = shares(a, Pa, R).set_index(a[key])
Y = shares(b, Pb, C).set_index(b[key])
w = a.set_index(key)['בזב']
common = X.index.intersection(Y.index)
X, Y, w = X.loc[common], Y.loc[common], w.loc[common].astype(float)
# keep units whose electorate changed less than 25% (stable composition)
e26 = b.set_index(key)['בזב'].loc[common]
ok = (e26 / w).between(.75, 1.33) & (w >= 100)
X, Y, w = X[ok].to_numpy(), Y[ok].to_numpy(), (w[ok] / w[ok].mean()).to_numpy()
print(f"{level}: {ok.sum()} units used")

Bm = cp.Variable((X.shape[1], Y.shape[1]), nonneg=True)
cp.Problem(cp.Minimize(cp.sum(cp.multiply(w[:, None], cp.square(X @ Bm - Y)))), [cp.sum(Bm, axis=1) == 1]).solve()
B = pd.DataFrame(Bm.value, index=list(R), columns=list(C)).clip(lower=0)
fit = X @ B.to_numpy()
r2 = 1 - (w[:, None] * (Y - fit) ** 2).sum() / (w[:, None] * (Y - np.average(Y, axis=0, weights=w)) ** 2).sum()
print(f"R2 {r2:.3f}")
pd.set_option("display.width", 250)
print((100 * B).round(0).astype(int).to_string())
B.to_csv(f"{DATA}/transfer_{level}.csv")

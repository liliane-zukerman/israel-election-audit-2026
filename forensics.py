#!/usr/bin/env python3
"""Forensic tests on Israel 26th Knesset results by ballot box (expb.csv from the Central Elections Committee)."""
import sys
import numpy as np
import pandas as pd
from scipy import stats

f = sys.argv[1] if len(sys.argv) > 1 else "dados/expb.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "dados/boxes.pkl"
b = pd.read_csv(f, encoding="utf-8-sig")
b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
meta = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים',
        'כשרים', 'חיצוניות', 'קלפי אחרת']
P = [c for c in b.columns if c not in meta]
b = b.rename(columns={'שם ישוב': 'city', 'סמל ישוב': 'city_id', 'קלפי': 'box', 'בזב': 'elig', 'מצביעים': 'voters',
                      'פסולים': 'invalid', 'כשרים': 'valid', 'שופט': 'judge'})
env = b[b.city_id == 9999]  # double-envelope boxes (soldiers, diplomats, hospitals...): no registered electorate
r = b[(b.city_id != 9999) & (b.elig > 0)].copy()
r["turnout"] = r.voters / r.elig
top = b[P].sum().sort_values(ascending=False).index[:11].tolist()
print(f"regular boxes {len(r)} | envelope boxes {len(env)} ({env.voters.sum()} voters)")

print("\n== 1. impossible values ==")
print("voters > eligible:", int((r.voters > r.elig).sum()), "| turnout >= 99%:", int((r.turnout >= .99).sum()),
      "| party sum != valid:", int((b[P].sum(axis=1) != b.valid).sum()),
      "| valid+invalid != voters:", int((b.valid + b.invalid != b.voters).sum()))
print(r.sort_values("turnout", ascending=False)[["city", "box", "elig", "voters", "turnout"]].head(12).round(3).to_string())


def smooth_expected(counts, digit_fn, ndig, base):
    """Expected digit frequencies from a smoothed histogram of the counts (removes the size effect)."""
    h = np.bincount(counts)
    k = 11
    sm = np.convolve(h, np.ones(k) / k, mode="same")
    exp = np.zeros(ndig)
    for v in range(len(sm)):
        d = digit_fn(v)
        if d is not None:
            exp[d - base] += sm[v]
    return exp / exp.sum()


def last_digit(v):
    return v % 10


def second_digit(v):
    return int(str(v)[1]) if v >= 10 else None


print("\n== 2. digit tests (chi-square vs smoothed-histogram baseline; naive baseline in brackets) ==")
benford2 = np.array([sum(np.log10(1 + 1 / (10 * d1 + d2)) for d1 in range(1, 10)) for d2 in range(10)])
rows = []
for col in ["voters", "invalid"] + top:
    x = r[col].to_numpy().astype(int)
    for name, sel, fn, naive in (("last", x >= 50, last_digit, np.full(10, .1)), ("2nd", x >= 10, second_digit, benford2)):
        v = x[sel]
        if len(v) < 300:
            continue
        obs = np.bincount([fn(i) for i in v], minlength=10)
        exp = smooth_expected(v, fn, 10, 0)
        p = stats.chisquare(obs, exp * obs.sum()).pvalue
        pn = stats.chisquare(obs, naive * obs.sum()).pvalue
        rows.append((col, name, len(v), round(p, 3), f"[{pn:.3g}]", (100 * obs / obs.sum()).round(1).tolist()))
print(pd.DataFrame(rows, columns=["series", "test", "boxes", "p", "p naive", "% by digit 0-9"]).to_string(index=False))

print("\n== 3. turnout: spikes at round percentages (Kobak test) ==")
t = 100 * r.turnout
for lo, hi in ((60, 100),):
    s = t[(t >= lo) & (t < hi)]
    integer = ((s - s.round()).abs() < 0.05).mean()
    # expected share within 0.05 of an integer, from uniform fractional parts
    print(f"boxes with turnout within 0.05pp of an integer: {100*integer:.2f}% (expected ~10%)")
    for target in (70, 75, 80, 85, 90, 95):
        print(f"  near {target}%: {((s - target).abs() < .05).sum()} | neighbours avg "
              f"{np.mean([((s - (target + o)).abs() < .05).sum() for o in (-2, -1, 1, 2)]):.1f}")

print("\n== 4. Klimek: turnout vs share of the box winner ==")
r["win_share"] = r[top].max(axis=1) / r.valid.clip(lower=1)
r["winner"] = r[top].idxmax(axis=1)
hi = r[(r.turnout >= .95) & (r.win_share >= .9)]
print(f"boxes with turnout>=95% and winner>=90%: {len(hi)}")
print(hi.groupby("winner").size().to_string())
for p in top[:6]:
    sh = r[p] / r.valid.clip(lower=1)
    within = np.corrcoef(r.turnout - r.groupby("city").turnout.transform("mean"),
                         sh - sh.groupby(r.city).transform("mean"))[0, 1]
    print(f"  corr(turnout, {p} share): national {np.corrcoef(r.turnout, sh)[0,1]:+.3f} | within city {within:+.3f}")
r.to_pickle(out)

print("\n== 5. invalid votes ==")
r["inv_rate"] = r.invalid / r.voters.clip(lower=1)
print(r.inv_rate.describe(percentiles=[.5, .9, .99, .999]).round(4).to_string())
print(r.sort_values("inv_rate", ascending=False)[["city", "box", "voters", "invalid", "inv_rate"]].head(10).round(3).to_string())

print("\n== 6. judge flag (column 'שופט') ==")
print(r.groupby("judge").agg(boxes=("box", "size"), turnout=("turnout", "mean"), inv=("inv_rate", "mean")).round(3))
print(r[r.judge == 1].city.value_counts().head(10).to_string())

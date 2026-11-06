#!/usr/bin/env python3
"""Double-envelope ballot boxes (locality code 9999: soldiers, diplomats, prisoners, hospitals, polling staff).

1. Vote shares in envelope boxes vs regular boxes, 2022 and 2026 (is the gap similar in both years?).
2. Digit tests on party counts in envelope boxes against a binomial simulation null (same method as sim_null.py).
3. Size and spread of envelope boxes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
YEARS = {"2022": "dados/expb_2022.csv", "2026": "dados/expb.csv"}
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
N = 500


def digits_p(x, n, fn, thr):
    """Parametric bootstrap p-value for a digit test: redraw counts as Binomial(n, observed share)."""
    share = x / np.clip(n, 1, None)

    def hist(v):
        d = fn(v[v >= thr])
        return np.bincount(d[d >= 0], minlength=10)[:10]
    obs = hist(x)
    sims = np.array([hist(rng.binomial(n, share)) for _ in range(N)])
    exp = sims.mean(axis=0)
    chi = lambda o: (((o - exp * o.sum() / exp.sum()) ** 2) / (exp * o.sum() / exp.sum())).sum()
    c = chi(obs)
    return obs.sum(), (1 + sum(chi(s) >= c for s in sims)) / (N + 1)


last = lambda v: v % 10
second = lambda v: np.where(v >= 10, np.where(v >= 100, (v // 10) % 10, v % 10), -1)

for year, f in YEARS.items():
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
    P = [c for c in b.columns if c not in META]
    env, reg = b[b['סמל ישוב'] == 9999], b[b['סמל ישוב'] != 9999]
    top = b[P].sum().sort_values(ascending=False).index[:11]
    print(f"\n===== {year}: envelope boxes {len(env)} | voters {env['מצביעים'].sum()} "
          f"({100 * env['מצביעים'].sum() / b['מצביעים'].sum():.1f}% of all) | median voters per box {env['מצביעים'].median():.0f}")
    print(f"invalid rate: envelopes {100 * env['פסולים'].sum() / env['מצביעים'].sum():.2f}% | regular "
          f"{100 * reg['פסולים'].sum() / reg['מצביעים'].sum():.2f}%")
    se, sr = 100 * env[top].sum() / env['כשרים'].sum(), 100 * reg[top].sum() / reg['כשרים'].sum()
    print(pd.DataFrame({"envelopes_%": se.round(2), "regular_%": sr.round(2), "diff_pp": (se - sr).round(2)}).to_string())
    v = env['מצביעים']
    print(f"envelope counts per box: median {v.median():.0f} | most common {v.value_counts().head(5).to_dict()} "
          "(boxes are filled in fixed batches, so the voters' digits are not tested)")
    print("digit tests on party counts in envelope boxes (bootstrap p):")
    for col in top[:6]:
        x, n = env[col].to_numpy(), env['כשרים'].to_numpy()
        for name, fn, thr in (("last", last, 50), ("2nd", second, 10)):
            k, p = digits_p(x, n, fn, thr)
            if k >= 200:
                print(f"  {col:5s} {name:4s} n={k:4d} p={p:.3f}")

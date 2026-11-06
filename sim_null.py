#!/usr/bin/env python3
"""Simulation nulls for the digit and round-turnout tests.

Digits: for each box, redraw the party count as Binomial(valid, observed share). This keeps the real
distribution of box sizes and shares and adds ordinary sampling noise. The chi-square of the observed
digits is compared with the chi-square of each simulated set (parametric bootstrap p-value).
Turnout: same idea with voters ~ Binomial(eligible, observed turnout), counting boxes whose turnout falls
exactly near each integer percentage (captures the 'simple fraction' effect, e.g. 3/4 = 75%).
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
r = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else "dados/boxes.pkl")
P = [c for c in r.columns if c not in ('סמל ועדה', 'ברזל', 'city', 'city_id', 'box', 'ריכוז', 'judge', 'elig', 'voters',
     'invalid', 'valid', 'חיצוניות', 'קלפי אחרת', 'turnout', 'win_share', 'winner', 'inv_rate')]
top = r[P].sum().sort_values(ascending=False).index[:11].tolist()
N = 500


def last(v):
    return v % 10


def second(v):
    s = np.where(v >= 100, (v // 10) % 10, v % 10)
    return np.where(v >= 10, s, -1)


def chi(obs, exp):
    return (((obs - exp) ** 2) / exp).sum()


rows = []
for col in ["voters", "invalid"] + top:
    n = r.valid.to_numpy() if col != "voters" else r.elig.to_numpy()
    if col == "invalid":
        n = r.voters.to_numpy()
    x = r[col].to_numpy()
    share = x / np.clip(n, 1, None)
    for name, fn, thr in (("last", last, 50), ("2nd", second, 10)):
        def hist(v):
            d = fn(v)[v >= thr]
            d = d[d >= 0]
            return np.bincount(d, minlength=10)[:10]
        obs = hist(x)
        if obs.sum() < 300:
            continue
        sims = np.array([hist(rng.binomial(n, share)) for _ in range(N)])
        exp = sims.mean(axis=0)
        c_obs = chi(obs, exp * obs.sum() / exp.sum())
        c_sim = np.array([chi(s, exp * s.sum() / exp.sum()) for s in sims])
        p = (1 + (c_sim >= c_obs).sum()) / (N + 1)
        rows.append((col, name, int(obs.sum()), round(p, 3)))
d = pd.DataFrame(rows, columns=["series", "test", "boxes", "p_sim"])
print(d.to_string(index=False))
print(f"\n{len(d)} tests; p<0.05: {(d.p_sim < .05).sum()} (expected by chance ~{.05*len(d):.1f}); p<0.01: {(d.p_sim < .01).sum()}")

# round-turnout spikes
t = 100 * r.turnout.to_numpy()
el, tu = r.elig.to_numpy(), r.turnout.to_numpy()
def near(tt, k):
    return int((np.abs(tt - k) < 0.05).sum())
sim_t = [100 * rng.binomial(el, tu) / el for _ in range(300)]
print("\nturnout near integer: observed vs simulated mean [p95]")
for k in range(60, 96, 5):
    s = np.array([near(st, k) for st in sim_t])
    print(f"  {k}%: {near(t, k)} vs {s.mean():.1f} [{np.percentile(s, 95):.0f}]")
tot_obs = sum(near(t, k) for k in range(50, 100))
tot_sim = np.array([sum(near(st, k) for k in range(50, 100)) for st in sim_t[:100]])
print(f"  all integers 50-99: {tot_obs} vs {tot_sim.mean():.0f} [p95 {np.percentile(tot_sim, 95):.0f}]")

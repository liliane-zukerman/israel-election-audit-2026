#!/usr/bin/env python3
"""Ballot boxes compared with the other boxes in the same polling site (column 'ריכוז' groups boxes by site).

Boxes in one site serve the same neighbourhood and voters are split between them (usually alphabetically),
so their results should differ only by ordinary sampling noise. For each box in a site with 3+ boxes:
  - chi-square of its party distribution against the pooled rest of the site (top parties + 'other')
  - z-score of its turnout against the rest of the site
The observed chi-square distribution is calibrated with a multinomial simulation under 'same site, same voters'.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

f = sys.argv[1] if len(sys.argv) > 1 else "dados/expb.csv"
rng = np.random.default_rng(9)
b = pd.read_csv(f, encoding="utf-8-sig")
b = b.loc[:, ~b.columns.str.startswith("Unnamed")]
META = ['סמל ועדה', 'ברזל', 'שם ישוב', 'סמל ישוב', 'קלפי', 'ריכוז', 'שופט', 'בזב', 'מצביעים', 'פסולים', 'כשרים',
        'חיצוניות', 'קלפי אחרת']
P = [c for c in b.columns if c not in META]
b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)].copy()
top = b[P].sum().sort_values(ascending=False).index[:9].tolist()
b["other"] = b[P].sum(axis=1) - b[top].sum(axis=1)
cats = top + ["other"]
b["site"] = b['סמל ישוב'].astype(str) + "/" + b['ריכוז'].astype(str)
b = b[b.groupby("site")["site"].transform("size") >= 3].copy()
print(f"boxes in sites with 3+ boxes: {len(b)} | sites: {b.site.nunique()}")

X = b[cats].to_numpy(float)
S = b.groupby("site")[cats].transform("sum").to_numpy(float)
rest = S - X
p_rest = rest / rest.sum(axis=1, keepdims=True)
n = X.sum(axis=1)


def chi_stat(X, n, p):
    e = n[:, None] * p
    ok = e > 0
    return np.where(ok, (X - e) ** 2 / np.where(ok, e, 1), 0).sum(axis=1)


obs = chi_stat(X, n, p_rest)
# simulation: every box drawn from its site's pooled distribution, recomputed exactly like the observed
site_idx = b.groupby("site").ngroup().to_numpy()
p_site = S / S.sum(axis=1, keepdims=True)
sim_max, sim_q = [], []
for _ in range(50):
    Xs = np.vstack([rng.multinomial(int(k), pp) for k, pp in zip(n, p_site)]).astype(float)
    Ss = pd.DataFrame(Xs).groupby(site_idx).transform("sum").to_numpy()
    rs = Ss - Xs
    ps = rs / np.clip(rs.sum(axis=1, keepdims=True), 1, None)
    cs = chi_stat(Xs, n, ps)
    sim_q.append(np.quantile(cs, [.5, .9, .99, .999]))
    sim_max.append(cs)
sim_all = np.concatenate(sim_max)
print("chi-square quantiles  observed:", np.quantile(obs, [.5, .9, .99, .999]).round(1),
      "| simulated:", np.mean(sim_q, axis=0).round(1))
thr = np.quantile(sim_all, 1 - 1e-4)
print(f"boxes above the simulated 1-in-10,000 threshold ({thr:.0f}): {(obs > thr).sum()} "
      f"(expected ~{len(obs) * 1e-4:.1f}) | over-dispersion ratio median {np.median(obs) / np.median(sim_all):.2f}")

b["chi"] = obs
b["p_sim"] = [(1 + (sim_all >= c).sum()) / (len(sim_all) + 1) for c in obs]
# turnout vs rest of site
E, V = b.groupby("site")['בזב'].transform("sum"), b.groupby("site")['מצביעים'].transform("sum")
tr = (V - b['מצביעים']) / (E - b['בזב'])
b["t"] = b['מצביעים'] / b['בזב']
b["z_turnout"] = (b.t - tr) / np.sqrt(tr * (1 - tr) / b['בזב'])
print(f"turnout |z| > 5 vs rest of site: {(b.z_turnout.abs() > 5).sum()} boxes")

flag = b.sort_values("chi", ascending=False).head(25)
show = []
for i, r in flag.iterrows():
    k = np.argmax(np.abs(X[b.index.get_loc(i)] / n[b.index.get_loc(i)] - p_rest[b.index.get_loc(i)]))
    show.append((r['שם ישוב'], r['קלפי'], r['ריכוז'], int(r['כשרים']), cats[k],
                 round(100 * X[b.index.get_loc(i), k] / n[b.index.get_loc(i)], 1), round(100 * p_rest[b.index.get_loc(i), k], 1),
                 round(r.chi), r.p_sim, round(r.z_turnout, 1)))
print("\nmost atypical boxes within their own site (party with largest gap shown):")
print(pd.DataFrame(show, columns=["locality", "box", "site", "valid", "party", "box_%", "rest_of_site_%", "chi2",
                                  "p_sim", "z_turnout"]).to_string(index=False))
b[["שם ישוב", "קלפי", "ריכוז", "כשרים", "chi", "p_sim", "z_turnout"]].to_csv("dados/same_site_results.csv", index=False)

# Likely transcription errors: a party that is strong in the rest of the site is (almost) missing in the box,
# while another list that is (almost) absent in the rest of the site appears with a similar number of votes.
Xdf = pd.DataFrame(X, index=b.index, columns=cats)
full = b[P]
Sfull = b.groupby("site")[P].transform("sum")
rest_full = (Sfull - full).div((Sfull - full).sum(axis=1).clip(lower=1), axis=0)
share_full = full.div(b['כשרים'].clip(lower=1), axis=0)
cand = []
for i in b.index:
    miss = [p for p in P if rest_full.at[i, p] >= .15 and share_full.at[i, p] <= .02]
    extra = [p for p in P if share_full.at[i, p] >= .08 and rest_full.at[i, p] <= .02]
    for a in miss:
        expected = rest_full.at[i, a] * b.at[i, 'כשרים']
        for e in extra:
            got = full.at[i, e]
            if got >= 20 and 0.5 <= got / max(expected, 1) <= 1.5:
                cand.append((b.at[i, 'שם ישוב'], b.at[i, 'קלפי'], a, round(100 * share_full.at[i, a], 1),
                             round(100 * rest_full.at[i, a], 1), e, int(got), round(expected)))
c = pd.DataFrame(cand, columns=["locality", "box", "missing_party", "box_%", "rest_of_site_%", "appears_as",
                                "votes_there", "votes_expected_for_missing"])
print(f"\nlikely transcription errors (votes of a party apparently recorded under another list's letters): {len(c)}")
print(c.sort_values("votes_there", ascending=False).to_string(index=False))
c.to_csv("dados/likely_transcription_errors.csv", index=False)

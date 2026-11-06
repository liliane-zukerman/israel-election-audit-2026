#!/usr/bin/env python3
"""Do invalid votes follow the local profile? Locality-level check with CBS municipal profiles (p_libud_2021.xlsx):
invalid-vote rate vs age 65+, recent immigrants, matriculation, higher education, socio-economic cluster, and whether
the association between invalid votes and Yisrael Beiteinu / Likud shares survives those controls."""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "data" if os.path.isdir("data") else "dados"
num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
p = pd.read_excel(f"{DATA}/p_libud_2021.xlsx", sheet_name='נתונים פיזיים ונתוני אוכלוסייה ', header=None).iloc[5:]
prof = pd.DataFrame({"code": num(p[1]), "name": p[0], "pop": num(p[12]), "arab": num(p[15]).fillna(0),
                     "age65": num(p[29]), "olim": num(p[43]), "olim1990": num(p[42]), "bagrut": num(p[152]), "acad": num(p[154]),
                     "ses": num(p[232]), "wage": num(p[115])}).dropna(subset=["code"])
prof["olim_per_1000"] = 1000 * prof.olim.fillna(0) / prof["pop"]

rows = []
for year, f, yb, lik in (("2022", f"{DATA}/expb_2022.csv", "ל", "מחל"), ("2026", f"{DATA}/expb.csv", "ל", "מחל")):
    b = pd.read_csv(f, encoding="utf-8-sig")
    b = b[(b['סמל ישוב'] != 9999) & (b['בזב'] > 0)]
    g = b.groupby('סמל ישוב').agg(voters=('מצביעים', 'sum'), inv=('פסולים', 'sum'), valid=('כשרים', 'sum'),
                                   yb=(yb, 'sum'), lik=(lik, 'sum')).reset_index()
    g["inv_rate"] = 100 * g.inv / g.voters
    g["yb_share"] = 100 * g.yb / g.valid
    g["lik_share"] = 100 * g.lik / g.valid
    d = g.merge(prof, left_on='סמל ישוב', right_on="code").dropna(subset=["age65", "bagrut", "ses"])
    d = d[d.voters >= 1000]
    print(f"  corr(Yisrael Beiteinu share, immigrants 1990+) = {np.corrcoef(d.yb_share, d.olim1990)[0, 1]:+.2f}")
    print(f"\n===== {year}: {len(d)} cities and local councils =====")
    for v in ("age65", "olim_per_1000", "olim1990", "bagrut", "acad", "ses", "arab", "yb_share", "lik_share"):
        print(f"  corr(invalid rate, {v:13s}) = {np.corrcoef(d.inv_rate, d[v])[0, 1]:+.2f}")
    m0 = smf.wls("inv_rate ~ yb_share + lik_share", d, weights=d.voters).fit()
    m1 = smf.wls("inv_rate ~ yb_share + lik_share + age65 + olim_per_1000 + bagrut + ses + arab", d, weights=d.voters).fit()
    m2 = smf.wls("inv_rate ~ yb_share + lik_share + age65 + olim1990 + bagrut + ses + arab", d, weights=d.voters).fit()
    m3 = smf.wls("inv_rate ~ olim1990 + age65 + bagrut + ses + arab", d, weights=d.voters).fit()
    print(f"  without profile: YB {m0.params.yb_share:+.4f} (p={m0.pvalues.yb_share:.3f}), Likud {m0.params.lik_share:+.4f} (p={m0.pvalues.lik_share:.3f}), R2 {m0.rsquared:.2f}")
    print(f"  with profile:    YB {m1.params.yb_share:+.4f} (p={m1.pvalues.yb_share:.3f}), Likud {m1.params.lik_share:+.4f} (p={m1.pvalues.lik_share:.3f}), R2 {m1.rsquared:.2f}")
    print(f"  with 1990+ olim: YB {m2.params.yb_share:+.4f} (p={m2.pvalues.yb_share:.3f}), Likud {m2.params.lik_share:+.4f} (p={m2.pvalues.lik_share:.3f}), olim1990 {m2.params.olim1990:+.4f} (p={m2.pvalues.olim1990:.3f}), R2 {m2.rsquared:.2f}")
    print(f"  olim1990 without party shares: {m3.params.olim1990:+.4f} (p={m3.pvalues.olim1990:.3f}), R2 {m3.rsquared:.2f}")
    print("  profile terms:", {k: f"{m1.params[k]:+.4f} (p={m1.pvalues[k]:.3f})" for k in ("age65", "olim_per_1000", "bagrut", "ses", "arab")})

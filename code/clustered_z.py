"""Clustered (by firm) z-statistics for the three sibling measures, bunchers vs just-above, pooled 2018-2023 and 2015-2017."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
exec(open(f"{ROOT}/scripts/splitting_test.py").read().split("rows = []")[0])
recs = []
for y in range(2015, 2024):
    d = p[p.year == y].copy(); d["n_sib"] = d.cui.map(n_sib).fillna(0)
    d["sib_new"] = d.cui.isin(flag_window(sib, y)); d["addr_new"] = d.cui.isin(flag_window(asib, y)); d["any_sib"] = d.n_sib > 0
    d["grp"] = np.select([(d.rel >= -0.05) & (d.rel < 0), (d.rel >= 0) & (d.rel < 0.10)], ["bunchers", "just_above"], default="other")
    recs.append(d[d.grp != "other"][["year", "cui", "grp", "sib_new", "addr_new", "any_sib"]])
S = pd.concat(recs); rows = []
for y0, y1 in ((2018, 2023), (2015, 2017)):
    T = S[(S.year >= y0) & (S.year <= y1)]
    for col in ["sib_new", "addr_new", "any_sib"]:
        a, b = T[T.grp == "bunchers"], T[T.grp == "just_above"]; pa, pb = a[col].mean(), b[col].mean()
        def var_cl(x, m): r = (x[col].astype(float) - m).groupby(x.cui).sum(); return (r ** 2).sum() / len(x) ** 2
        z = (pa - pb) / np.sqrt(var_cl(a, pa) + var_cl(b, pb)); rows.append(dict(years=f"{y0}--{y1}", measure=col, bunchers=round(100 * pa, 2), just_above=round(100 * pb, 2), z_clustered=round(z, 1)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{OUT}/splitting_clustered_z.csv", index=False)

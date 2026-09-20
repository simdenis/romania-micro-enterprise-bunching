"""Fourth round.
A. Margin event study: firms moved from profit tax to the micro regime by the 2018 ceiling increase (2017 revenue in
   (500k, 1M] EUR) vs firms just above (1M, 2M] that stayed on profit tax; reported pre-tax margin 2015-2020.
   Symmetric event in 2023: firms with 2022 revenue in (500k, 1M] lost the regime in 2023; control (1M, 2M]. Years 2020-2025.
   Placebo: bands on 2019 revenue, years 2017-2021 (no regime change for either band).
B. Combined-revenue bunching in 2024: sums of revenue over firms sharing an administrator, around EUR 500k at the
   end-2023 rate, vs the same object in 2021-2023 (aggregation rule only from 2024).
C. Sector heterogeneity of the DiB excess, by CAEN section, for the 2021, 2023 and 2025 notches.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "caen", "revenue", "costs", "pretax", "employees"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")[["year", "cui", "regime"]]
pp = pp.merge(reg, on=["year", "cui"], how="left"); pp["micro"] = pp.regime.isin(["micro", "micro_loss"])
pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1); pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y])   # end-of-same-year rate for year-end status tests

# ---------- A. event studies ----------
def event(base_year, lo, hi, years, label):
    b = pp[(pp.year == base_year) & (pp.revenue > 0)]
    T = set(b[(b.rev_eur > lo) & (b.rev_eur <= hi)].cui); C = set(b[(b.rev_eur > hi) & (b.rev_eur <= 2 * hi)].cui)
    d = pp[pp.year.isin(years) & (pp.revenue > 0)]
    # balanced: present in all years
    cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index); T &= keep; C &= keep
    rows = []
    for y in years:
        dt, dc = d[(d.year == y) & d.cui.isin(T)], d[(d.year == y) & d.cui.isin(C)]
        rows.append(dict(design=label, year=y, n_treated=len(dt), n_control=len(dc), micro_share_T=round(100 * dt.micro.mean(), 1), micro_share_C=round(100 * dc.micro.mean(), 1),
                         median_margin_T=round(dt.margin.median(), 4), median_margin_C=round(dc.margin.median(), 4), mean_margin_T=round(dt.margin.mean(), 4), mean_margin_C=round(dc.margin.mean(), 4),
                         median_rev_T_keur=round(dt.rev_eur.median() / 1e3), median_rev_C_keur=round(dc.rev_eur.median() / 1e3),
                         share_loss_T=round(100 * (dt.pretax <= 0).mean(), 1), share_loss_C=round(100 * (dc.pretax <= 0).mean(), 1)))
    r = pd.DataFrame(rows); base = r[r.year == base_year].iloc[0]
    r["DiD_median_margin"] = ((r.median_margin_T - base.median_margin_T) - (r.median_margin_C - base.median_margin_C)).round(4)
    r["DiD_mean_margin"] = ((r.mean_margin_T - base.mean_margin_T) - (r.mean_margin_C - base.mean_margin_C)).round(4)
    return r
A = pd.concat([event(2017, 500e3, 1e6, list(range(2015, 2021)), "2018: (500k,1M] moved to micro vs (1M,2M]"),
               event(2022, 500e3, 1e6, list(range(2020, 2026)), "2023: (500k,1M] lost micro vs (1M,2M]"),
               event(2019, 500e3, 1e6, list(range(2017, 2022)), "placebo: bands on 2019 revenue")])
pd.set_option("display.width", 260); print("A. Event studies (balanced panels; margins = pre-tax profit / revenue):\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev4_margin_event_study.csv", index=False)

# ---------- B. combined-revenue bunching ----------
link = pd.read_parquet(f"{ROOT}/data/panel/onrc_person_firm.parquet"); adm = pd.read_parquet(f"{ROOT}/data/panel/onrc_admins.parquet")
link = link[link.person_id.isin(set(adm.loc[adm.has_birth, "person_id"]))].dropna(subset=["cui"])
firms = pd.read_parquet(f"{ROOT}/data/panel/onrc_firms.parquet", columns=["cui", "FORMA_JURIDICA"]); srl = set(firms[firms.FORMA_JURIDICA.isin(["SRL", "SRL-D"])].cui)
link = link[link.cui.isin(srl)]
npf = link.groupby("person_id").cui.nunique(); multi = set(npf[(npf >= 2) & (npf <= 5)].index); link_m = link[link.person_id.isin(multi)]
grp = {}
for y in range(2019, 2026):
    d = pp[(pp.year == y) & (pp.revenue > 0)][["cui", "revenue"]].merge(link_m[["person_id", "cui"]], on="cui")
    g = d.groupby("person_id").agg(n=("cui", "nunique"), total=("revenue", "sum"), maxrev=("revenue", "max"))
    grp[y] = g[g.n >= 2]
def rel_hist_arr(v, loc, bw=0.01, half=0.20):
    edges = loc * (1 + np.arange(-half, half + bw / 2, bw)); c, _ = np.histogram(v, bins=edges); return c.astype(float)
rows = []
for y in (2021, 2022, 2023, 2024, 2025):
    loc = 500e3 * fx[y - 1]; v = np.sort(grp[y].total.values)
    q = np.searchsorted(v, loc) / len(v)
    ctr = [c for c in (2019, 2020, 2021, 2022) if c != y]     # years before any aggregation rule; individual notch at 1M
    cvals = []
    for c in ctr:
        vc = np.sort(grp[c].total.values); lc = vc[min(int(q * len(vc)), len(vc) - 1)]
        if abs(1e6 * fx[c - 1] / lc - 1) <= 0.25: continue                     # skip if the individual 1M notch is near the matched location
        cvals.append(rel_hist_arr(vc, lc))
    if not cvals: continue
    ct = rel_hist_arr(v, loc); st = ct / ct.sum(); sc = np.mean([c / c.sum() for c in cvals], axis=0)
    b = (st[15:20] - sc[15:20]).sum() / sc[15:20].mean(); exc = (st[15:20] - sc[15:20]).sum() * ct.sum()
    # individual firms inside multi-firm groups vs singletons: share within 5% below the individual notch that year
    Ly = MICRO[y] * fx[y - 1]; d = pp[(pp.year == y) & (pp.revenue > 0)]; in_grp = d.cui.isin(set(link_m.cui))
    def share_bunch(x): return 100 * ((x.revenue >= 0.95 * Ly) & (x.revenue < Ly)).sum() / ((x.revenue >= 0.8 * Ly) & (x.revenue < 1.2 * Ly)).sum()
    rows.append(dict(year=y, groups=len(v), location_eur=500e3, controls=",".join(map(str, [c for c in ctr])), groups_in_window=int(ct.sum()), b_group_total=round(b, 2), excess_groups=int(round(exc)),
                     firm_level_bunch_share_in_groups=round(share_bunch(d[in_grp]), 2), firm_level_bunch_share_singletons=round(share_bunch(d[~in_grp]), 2)))
B = pd.DataFrame(rows); print("\nB. Bunching of combined revenue of firms sharing an administrator, at EUR 500k (aggregation rule from 2024):\n" + B.to_string(index=False)); B.to_csv(f"{OUT}/rev4_group_bunching.csv", index=False)

# ---------- C. sector heterogeneity ----------
SEC = [("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D-E", 35, 39), ("F", 41, 43), ("G", 45, 47), ("H", 49, 53), ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68), ("M", 69, 75), ("N", 77, 82), ("P", 85, 85), ("Q", 86, 88), ("R-S", 90, 96)]
NAMES = {"A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D-E": "Utilities", "F": "Construction", "G": "Trade", "H": "Transport", "I": "Hospitality", "J": "Information and communication", "K": "Finance", "L": "Real estate", "M": "Professional services", "N": "Administrative services", "P": "Education", "Q": "Health", "R-S": "Arts, other services"}
pp["div"] = (pp.caen.fillna(0).astype(int) // 100)
def section(dv):
    for s, lo, hi in SEC:
        if lo <= dv <= hi: return s
    return None
pp["sec"] = pp["div"].map(section)
rows = []
for y in (2021, 2023, 2025):
    loc = MICRO[y] * fx[y - 1]; q_all = quantile_of(y, loc)
    ctr = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q_all) - 1) > 0.25 for n in notches(c))]
    for s, _, _ in SEC:
        vt = np.sort(pp[(pp.year == y) & (pp.revenue > 0) & (pp.sec == s)].revenue.values)
        if len(vt) == 0: continue
        ct = rel_hist_arr(vt, loc)
        if ct.sum() < 800: continue
        qs = np.searchsorted(vt, loc) / len(vt); cvals = []
        for c in ctr:
            vc = np.sort(pp[(pp.year == c) & (pp.revenue > 0) & (pp.sec == s)].revenue.values); lc = vc[min(int(qs * len(vc)), len(vc) - 1)]
            cvals.append(rel_hist_arr(vc, lc))
        st = ct / ct.sum(); sc = np.mean([c / c.sum() for c in cvals], axis=0)
        b = (st[15:20] - sc[15:20]).sum() / sc[15:20].mean(); exc = (st[15:20] - sc[15:20]).sum() * ct.sum()
        marg = pp[(pp.year == y) & (pp.revenue > 0) & (pp.sec == s)]; rel = marg.revenue / loc - 1
        rows.append(dict(year=y, section=s, name=NAMES[s], firms_in_window=int(ct.sum()), b=round(b, 2), excess_firms=int(round(exc)), excess_share_pct=round(100 * exc / ct.sum(), 2),
                         median_margin_bunchers=round(marg.margin[(rel >= -0.05) & (rel < 0)].median(), 3)))
C = pd.DataFrame(rows); print("\nC. Sector heterogeneity (DiB within CAEN section, same control years as the pooled estimate):\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/rev4_sector.csv", index=False)

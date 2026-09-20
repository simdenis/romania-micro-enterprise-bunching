"""Second revision round.
A. Economic quantities per notch: B/N_t (excess as share of all filers), hole width above the notch by the
   convergence method (bins until the cumulative deficit above offsets the excess below), and a reduced-form
   elasticity e = (dz*/L)^2 / (2 * dt / (1 - tau)), dt = 0.16*margin - tau evaluated at the median margin of bunchers.
B. Quantile contamination: iterate q <- F_t(L) - B/N_t and report the change in b.
C. Robustness: windows +/-10/15/30%; control subsets (pre-only, post-only, leave-one-year-out range); continuing firms
   (present in t-1 and t+1) only; 2023 without hospitality (CAEN 55-56); year-block jackknife SE.
D. 2023 decomposition: excess mass among first-time filers / firms aged <= 1 vs incumbents.
E. 1,000-lei modal bins with counts, 2018-2022.
F. Splitting: clustered z (cluster = firm), excluding persons without birth date; share of SRLs with one administrator.
G. VAT 2025 at 395,000 lei (polynomial) for the table.
"""
import os, itertools, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]
exec(compile(src, "dib", "exec"))
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "caen", "revenue", "pretax", "turnover"])
RATE = {2014: .03, 2015: .03, 2016: .03, 2017: .01, 2018: .01, 2019: .01, 2020: .01, 2021: .01, 2022: .01, 2023: .01, 2024: .03, 2025: .03}  # rate a buncher pays (3% no-employee 2014-16; 1% typical 2017-23; 3% above 60k 2024-25)
N = {y: len(rev[y]) for y in rev}

def controls_for(y, q, years=range(2014, 2026)):
    return [c for c in years if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]

def hole_width(st, sc, bw=0.01, half=0.20, region=0.05):
    k0 = int(round(half / bw)); nb = int(round(region / bw)); excess = (st[k0 - nb:k0] - sc[k0 - nb:k0]).sum()
    cum = 0.0
    for j in range(k0, len(st)):
        cum += sc[j] - st[j]
        if cum >= excess: return (j - k0 + 1) * bw
    return np.nan

# ---------- A + B ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q); b, m, st, sc, ct, ccs = estimate(y, loc, ctr)
    B_share = (st[15:20] - sc[15:20]).sum() * ct.sum() / N[y]
    dz = hole_width(st, sc)
    d = pp[(pp.year == y) & (pp.revenue > 0)]; rel = d.revenue / loc - 1
    margin = (d.pretax / d.revenue)[(rel >= -0.05) & (rel < 0)].median()
    tau = RATE[y]; dt = max(0.16 * margin - tau, 1e-4); e = (dz ** 2) / (2 * dt / (1 - tau)) if dz == dz else np.nan
    # quantile contamination: shift q down by B/N and re-estimate
    q2 = q - B_share; loc2 = value_at(y, q2)
    b2 = estimate(y, loc, controls_for(y, q2))[0] if True else np.nan   # same location, controls matched at corrected quantile
    ccs2 = [rel_hist(c, value_at(c, q2), 0.01, 0.20) for c in controls_for(y, q2)]; sc2 = np.mean([c / c.sum() for c in ccs2], axis=0)
    b2 = (st[15:20] - sc2[15:20]).sum() / sc2[15:20].mean()
    rows.append(dict(notch=label, year=y, b=round(b, 2), B_over_N_pct=round(100 * B_share, 3), hole_width_pct=round(100 * dz, 1) if dz == dz else np.nan,
                     median_margin_bunchers=round(margin, 3), tau=tau, dt_avg_rate_change=round(dt, 3), elasticity=round(e, 3) if e == e else np.nan,
                     b_quantile_corrected=round(b2, 2)))
A = pd.DataFrame(rows); pd.set_option("display.width", 250); print("A/B. Economic quantities and quantile correction:\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev2_quantities.csv", index=False)

# ---------- C. robustness ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q); base = estimate(y, loc, ctr)[0]
    r = dict(notch=label, b_base=round(base, 2))
    for half in (0.10, 0.15, 0.30):
        r[f"b_win{int(half*100)}"] = round(estimate(y, loc, ctr, half=half)[0], 2)
    pre = [c for c in ctr if c < y]; post = [c for c in ctr if c > y]
    r["b_pre_only"] = round(estimate(y, loc, pre)[0], 2) if pre else np.nan; r["b_post_only"] = round(estimate(y, loc, post)[0], 2) if post else np.nan
    loo = [estimate(y, loc, [c for c in ctr if c != k])[0] for k in ctr] if len(ctr) > 1 else [base]
    r["b_loo_min"], r["b_loo_max"] = round(min(loo), 2), round(max(loo), 2)
    r["jackknife_se"] = round(np.sqrt((len(ctr) - 1) / len(ctr) * sum((v - np.mean(loo)) ** 2 for v in loo)), 2) if len(ctr) > 1 else np.nan
    # continuing firms: present with revenue>0 in t-1 and t+1 (where available)
    ids = set(pp[(pp.year == y) & (pp.revenue > 0)].cui)
    for yy in (y - 1, y + 1):
        if yy in rev: ids &= set(pp[(pp.year == yy) & (pp.revenue > 0)].cui)
    sub = np.sort(pp[(pp.year == y) & pp.cui.isin(ids)].revenue.values)
    edges = loc * (1 + np.arange(-0.20, 0.20 + 0.005, 0.01)); ct_c, _ = np.histogram(sub, bins=edges); st_c = ct_c / ct_c.sum()
    sc_c = np.mean([(lambda cc: cc / cc.sum())(rel_hist(c, value_at(c, q), 0.01, 0.20)) for c in ctr], axis=0)
    r["b_continuing"] = round((st_c[15:20] - sc_c[15:20]).sum() / sc_c[15:20].mean(), 2); r["n_continuing"] = int(ct_c.sum())
    rows.append(r)
C = pd.DataFrame(rows); print("\nC. Robustness:\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/rev2_robustness.csv", index=False)

# 2023 without hospitality
d23 = pp[(pp.year == 2023) & (pp.revenue > 0)]; horeca = d23.caen.between(5500, 5699)
loc = 500e3 * fx[2022]; q = quantile_of(2023, loc); ctr = controls_for(2023, q)
for name, sel in [("all", d23), ("excluding CAEN 55-56", d23[~horeca]), ("CAEN 55-56 only", d23[horeca])]:
    edges = loc * (1 + np.arange(-0.20, 0.20 + 0.005, 0.01)); ct_h, _ = np.histogram(sel.revenue.values, bins=edges); st_h = ct_h / ct_h.sum()
    sc_h = np.mean([(lambda cc: cc / cc.sum())(rel_hist(c, value_at(c, q), 0.01, 0.20)) for c in ctr], axis=0)
    print(f"   2023 {name}: firms in window {int(ct_h.sum()):,}, b = {(st_h[15:20] - sc_h[15:20]).sum() / sc_h[15:20].mean():.2f}")

# ---------- D. 2023 decomposition ----------
firms = pd.read_parquet(f"{ROOT}/data/panel/onrc_firms.parquet", columns=["cui", "reg_date"]).set_index("cui").reg_date
first = pp[pp.revenue > 0].groupby("cui").year.min()
for y, loc in [(2023, 500e3 * fx[2022]), (2021, 1e6 * fx[2020]), (2024, 500e3 * fx[2023])]:
    d = pp[(pp.year == y) & (pp.revenue > 0)].copy(); d["rel"] = d.revenue / loc - 1
    d["new"] = (first.reindex(d.cui).values == y) | ((y - firms.reindex(d.cui).dt.year.values) <= 1)
    q = quantile_of(y, loc); ctr = controls_for(y, q)
    sc_d = np.mean([(lambda cc: cc / cc.sum())(rel_hist(c, value_at(c, q), 0.01, 0.20)) for c in ctr], axis=0)
    edges = loc * (1 + np.arange(-0.20, 0.20 + 0.005, 0.01)); ct_all, _ = np.histogram(d.revenue.values, bins=edges)
    ct_new, _ = np.histogram(d[d.new].revenue.values, bins=edges); ct_old = ct_all - ct_new
    exc_all = ((ct_all / ct_all.sum())[15:20] - sc_d[15:20]).sum() * ct_all.sum()
    # share of the excess attributable to new firms: compare new-firm share in bunching bins vs in the rest of the window
    share_new_bunch = ct_new[15:20].sum() / ct_all[15:20].sum(); share_new_rest = (ct_new.sum() - ct_new[15:20].sum()) / (ct_all.sum() - ct_all[15:20].sum())
    excess_new = (share_new_bunch - share_new_rest) * ct_all[15:20].sum()
    print(f"D. {y}: excess firms {exc_all:,.0f}; new firms (first filing or age<=1) are {100*share_new_bunch:.1f}% of the bunching bins vs {100*share_new_rest:.1f}% of the rest of the window -> excess new firms ~{excess_new:,.0f} ({100*excess_new/exc_all:.0f}% of excess)")

# ---------- E. 1,000-lei modal bins ----------
rows = []
for y in range(2018, 2023):
    v = rev[y]; tp, tc = 1e6 * fx[y - 1], 1e6 * fx[y]; lo = min(tp, tc) - 60_000; hi = max(tp, tc) + 60_000
    edges = np.arange(np.floor(lo / 1000) * 1000, hi + 1000, 1000); c, _ = np.histogram(v, bins=edges); k = np.argmax(c)
    # also the 5,000-lei bin
    e5 = np.arange(np.floor(lo / 5000) * 5000, hi + 5000, 5000); c5, _ = np.histogram(v, bins=e5); k5 = np.argmax(c5)
    rows.append(dict(year=y, thr_prev=round(tp), thr_cur=round(tc), modal_1k_bin=f"{int(edges[k]):,}-{int(edges[k+1]):,}", firms_1k=int(c[k]), median_1k=int(np.median(c)),
                     dist_1k_to_prev=int(edges[k] + 500 - tp), modal_5k_bin=f"{int(e5[k5]):,}-{int(e5[k5+1]):,}", firms_5k=int(c5[k5]), dist_5k_to_prev=int(e5[k5] + 2500 - tp)))
E = pd.DataFrame(rows); print("\nE. Fine-bin modal locations:\n" + E.to_string(index=False)); E.to_csv(f"{OUT}/rev2_lei_modal_fine.csv", index=False)

# ---------- G. VAT 2025 at 395k ----------
def poly_b(values, thr, bw=2000, half=0.4, excl=3, deg=5):
    edges = np.arange(thr * (1 - half), thr * (1 + half) + bw, bw); c, _ = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], bins=edges); c = c.astype(float)
    centers = edges[:-1] + bw / 2; x = (centers - thr) / bw; mask = (x < -excl) | (x > excl); cf = np.polyval(np.polyfit(x[mask], c[mask], deg), x); below = (x >= -excl) & (x < 0)
    return (c[below].sum() - cf[below].sum()) / cf[below].mean(), int(c.sum())
t25 = pp[(pp.year == 2025) & (pp.turnover > 0)].turnover.values; t24 = pp[(pp.year == 2024) & (pp.turnover > 0)].turnover.values
b395, n395 = poly_b(t25, 395_000); b395p, n395p = poly_b(t24, 395_000); b300, n300 = poly_b(t25, 300_000)
print(f"\nG. VAT 2025: b at 395,000 lei = {b395:.2f} (n={n395:,}); 2024 placebo at 395,000 = {b395p:.2f}; 2025 at 300,000 = {b300:.2f}")
pd.DataFrame([dict(year=2025, vat_threshold_lei=395_000, firms_in_window=n395, excess_mass_b=round(b395, 2)), dict(year=2024, vat_threshold_lei=395_000, firms_in_window=n395p, excess_mass_b=round(b395p, 2))]).to_csv(f"{OUT}/rev2_vat_395k.csv", index=False)

# ---------- F. splitting: clustered z, no-birthdate exclusion, single-admin share ----------
exec(open(f"{ROOT}/scripts/splitting_test.py").read().split("rows = []")[0])
adm = pd.read_parquet(f"{ROOT}/data/panel/onrc_admins.parquet"); nb_persons = set(adm.loc[~adm.has_birth, "person_id"])
link_nb = link[~link.person_id.isin(nb_persons)]
pf2 = link_nb[["person_id", "code", "cui", "reg_date"]].dropna(subset=["reg_date"]).copy(); pf2["pid"] = pd.factorize(pf2.person_id)[0]; pf2["fid"] = pd.factorize(pf2.code)[0]
npp2 = pf2.groupby("pid").fid.transform("size"); pf2 = pf2[(npp2 >= 2) & (npp2 <= 50)][["pid", "fid", "cui", "reg_date"]]
sib2 = pf2.merge(pf2[["pid", "fid", "reg_date"]], on="pid", suffixes=("", "_sib")); sib2 = sib2[(sib2.fid != sib2.fid_sib) & sib2.cui.notna()][["cui", "fid_sib", "reg_date_sib"]].drop_duplicates()
recs = []
for y in range(2018, 2024):
    d = p[p.year == y].copy(); d["sib_new"] = d.cui.isin(flag_window(sib, y)); d["sib_new_nb"] = d.cui.isin(flag_window(sib2, y)); d["addr_new"] = d.cui.isin(flag_window(asib, y))
    d["grp"] = np.select([(d.rel >= -0.05) & (d.rel < 0), (d.rel >= 0) & (d.rel < 0.10)], ["bunchers", "just_above"], default="other"); recs.append(d[d.grp != "other"][["year", "cui", "grp", "sib_new", "sib_new_nb", "addr_new"]])
S = pd.concat(recs)
def clustered_z(S, col):
    a, b = S[S.grp == "bunchers"], S[S.grp == "just_above"]; pa, pb = a[col].mean(), b[col].mean()
    def var_cl(x):   # cluster-robust variance of a mean, clusters = firms
        g = x.groupby("cui")[col].agg(["sum", "size"]); resid = g["sum"] - pa_ * g["size"] if False else None
        m = x[col].mean(); r = (x[col] - m).groupby(x.cui).sum(); return (r ** 2).sum() / len(x) ** 2
    va, vb = var_cl(a), var_cl(b); return pa * 100, pb * 100, (pa - pb) / np.sqrt(va + vb), a.cui.nunique(), len(a)
for col in ["sib_new", "sib_new_nb", "addr_new"]:
    pa, pb, z, ncl, nobs = clustered_z(S, col); print(f"F. {col:11} pooled 2018-23: bunchers {pa:.2f}% vs just above {pb:.2f}%  clustered z = {z:.1f}  (bunchers: {nobs:,} firm-years, {ncl:,} firms)")
nadm = link.groupby("cui").person_id.nunique(); srl_ids = p.cui.unique()
print(f"   SRLs in the analysis sample with exactly one linked administrator: {100 * (nadm.reindex(srl_ids) == 1).mean():.1f}%")

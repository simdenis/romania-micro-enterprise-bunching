"""Eighth round.
A. Margins by position measured in t-1 (before the in-year crossers switched base), with the t-1 micro share of both groups.
B. Difference-in-bunching within lagged-margin groups (t-1 margin above / at or below the break-even margin for the rate at the ceiling).
C. First stage of the 2023 exit design from the ANAF registry flags (Aug 2023 and June 2024 snapshots) instead of the accounts inference.
D. Count-based DiB with the window extended to +/-50%: treated minus counterfactual firms in 10% slices.
E. Growth of firms 5-20% below the notch from t to t+1 in notch years against the same quantile band in control years, and against
   firms 20-35% below in the same year.
F. Excess firms in the 5% region for windows of 10, 20 and 30% (count version of the window check).
G. 2025 residual spike at 296-300k lei by regime and by listed activity (the 60k rate-boundary test).
H. The 2015 year-end notch (EUR 100k for 2016 status at the end-2015 rate): 2,000-lei histogram 2014-2016 and margins by position.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pd.set_option("display.width", 260)
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax", "tax", "caen"]).drop_duplicates(["year", "cui"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")[["year", "cui", "regime"]]
pp = pp.merge(reg, on=["year", "cui"], how="left"); pp["micro"] = pp.regime.isin(["micro", "micro_loss"]); pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1)
pp = pp[pp.revenue > 0].copy()
lag = pp[["year", "cui", "margin", "micro", "revenue"]].rename(columns={"margin": "margin_lag", "micro": "micro_lag", "revenue": "revenue_lag"}); lag["year"] = lag.year + 1
pp = pp.merge(lag, on=["year", "cui"], how="left")
N = {y: len(rev[y]) for y in rev}
CUT = {y: (0.1875 if y in (2014, 2015, 2024, 2025) else 0.0625) for y in range(2014, 2026)}      # break-even margin for the rate paid at the ceiling
def controls_for(y, q): return [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
MAIN = [(l, y, loc) for l, y, loc in TARGETS if "micro" in l]

# ---------- A ----------
rows = []
for y in range(2015, 2026):
    L = MICRO[y] * fx[y - 1]; d = pp[pp.year == y].copy(); d["rel"] = d.revenue / L - 1
    for g, m in [("bunchers", (d.rel >= -0.05) & (d.rel < 0)), ("just_above", (d.rel >= 0) & (d.rel < 0.10))]:
        x = d[m]; xl = x.dropna(subset=["margin_lag"])
        rows.append(dict(year=y, group=g, n=len(x), n_with_lag=len(xl), median_margin_t=round(x.margin.median(), 3), median_margin_lag=round(xl.margin_lag.median(), 3),
                         share_lag_gt_6_25=round(100 * (xl.margin_lag > 0.0625).mean(), 1), share_lag_gt_18_75=round(100 * (xl.margin_lag > 0.1875).mean(), 1),
                         share_t_gt_cut=round(100 * (x.margin > CUT[y]).mean(), 1), share_lag_gt_cut=round(100 * (xl.margin_lag > CUT[y]).mean(), 1),
                         micro_share_t=round(100 * x.micro.mean(), 1), micro_share_lag=round(100 * xl.micro_lag.mean(), 1), lag_below_ceiling=round(100 * (xl.revenue_lag < L).mean(), 1)))
A = pd.DataFrame(rows); print("A. Margins by position, current year and t-1:\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev8_margins_lagged.csv", index=False)

# ---------- B ----------
def hist_arr(v, loc, bw=0.01, half=0.20):
    edges = loc * (1 + np.arange(-half, half + bw / 2, bw)); c, _ = np.histogram(v, bins=edges); return c.astype(float)
rows = []
for label, y, loc in MAIN:
    if y == 2014: continue                                             # no t-1 statement
    q = quantile_of(y, loc); ctr = [c for c in controls_for(y, q) if c >= 2015]; cut = CUT[y]
    for g, cond in (("gain", lambda d: d.margin_lag > cut), ("no gain", lambda d: d.margin_lag <= cut)):
        arr = {c: np.sort(pp[(pp.year == c) & cond(pp)].revenue.values) for c in [y] + ctr}
        qg = np.searchsorted(arr[y], loc) / len(arr[y]); ct = hist_arr(arr[y], loc); st = ct / ct.sum()
        ccs = [hist_arr(arr[c], arr[c][min(int(qg * len(arr[c])), len(arr[c]) - 1)]) for c in ctr]; sc = np.mean([cc / cc.sum() for cc in ccs], axis=0)
        b = (st[15:20] - sc[15:20]).sum() / sc[15:20].mean(); exc = (st[15:20] - sc[15:20]).sum() * ct.sum()
        se = bootstrap(ct, ccs, 0.01, 0.05, 0.20, reps=200)
        rows.append(dict(notch=label, cut_pct=round(100 * cut, 2), group=g, firms_in_window=int(ct.sum()), b=round(b, 2), se=round(se, 2), excess_firms=int(round(exc)), excess_pct_of_window=round(100 * exc / ct.sum(), 1),
                         group_share_of_year=round(100 * len(arr[y]) / N[y], 1)))
B = pd.DataFrame(rows); print("\nB. DiB within lagged-margin groups:\n" + B.to_string(index=False)); B.to_csv(f"{OUT}/rev8_dib_by_margin.csv", index=False)

# ---------- C ----------
def anaf_flags(path):
    a = pd.read_csv(path, sep="|", encoding="utf-8-sig", usecols=["COD_FISCAL", "IMP100", "IMP120"], dtype=str, on_bad_lines="skip", engine="c", quoting=3, index_col=False)
    a["cui"] = pd.to_numeric(a.COD_FISCAL, errors="coerce"); a = a.dropna(subset=["cui"]).drop_duplicates("cui"); a["cui"] = a.cui.astype("int64")
    a["m"] = (a.IMP120.str.strip() == "DA") & (a.IMP100.str.strip() != "DA"); a["p"] = (a.IMP100.str.strip() == "DA") & (a.IMP120.str.strip() != "DA")
    return a.set_index("cui")[["m", "p"]]
F23 = anaf_flags(f"{ROOT}/data/raw/anaf/date_identificare_platitori_2023.csv"); F24 = anaf_flags(f"{ROOT}/data/raw/anaf/anaf2024/date_identificare_platitori_2024.csv")
pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y])
def bands(base, years, cut, lo_t=0.8, hi_t=1.0, lo_c=1.0, hi_c=1.2):
    d = pp[pp.year.isin(years)]; cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index)
    a = pp[pp.year.isin([base - 1, base]) & pp.cui.isin(keep)].groupby("cui").rev_eur.mean()
    return set(a[(a > lo_t * cut) & (a <= hi_t * cut)].index), set(a[(a > lo_c * cut) & (a <= hi_c * cut)].index)
rows = []
for ctrl_lab, lo_c, hi_c in (("control (1.0,1.2]", 1.0, 1.2), ("control (1.2,1.5]", 1.2, 1.5)):
    T, C = bands(2022, list(range(2020, 2026)), 1e6, lo_c=lo_c, hi_c=hi_c)
    for grp, S in (("treated", T), ("control", C)):
        ids = pd.Index(sorted(S))
        for snap, F, yr in (("Aug 2023", F23, 2023), ("Jun 2024", F24, 2024)):
            f = F.reindex(ids); inf = pp[(pp.year == yr) & pp.cui.isin(S)].micro.mean()
            rows.append(dict(control=ctrl_lab, group=grp, snapshot=snap, n=len(ids), matched_pct=round(100 * f.m.notna().mean(), 1), anaf_micro_pct=round(100 * f.m.fillna(False).mean(), 1),
                             anaf_profit_pct=round(100 * f.p.fillna(False).mean(), 1), inferred_micro_pct=round(100 * inf, 1)))
        inf22 = pp[(pp.year == 2022) & pp.cui.isin(S)].micro.mean(); rows.append(dict(control=ctrl_lab, group=grp, snapshot="(2022 base, inference only)", n=len(ids), matched_pct=np.nan, anaf_micro_pct=np.nan, anaf_profit_pct=np.nan, inferred_micro_pct=round(100 * inf22, 1)))
Cc = pd.DataFrame(rows); print("\nC. 2023 exit design: ANAF-flag micro shares vs inference:\n" + Cc.to_string(index=False)); Cc.to_csv(f"{OUT}/rev8_firststage_anaf.csv", index=False)

# ---------- D ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q)
    st = rel_hist(y, loc, 0.01, 0.50) / N[y]; sc = np.mean([rel_hist(c, value_at(c, q), 0.01, 0.50) / N[c] for c in ctr], axis=0); diff = (st - sc) * N[y]; cf = sc * N[y]
    sl = {"-50..-20": (0, 30), "-20..-5": (30, 45), "-5..0": (45, 50), "0..10": (50, 60), "10..20": (60, 70), "20..30": (70, 80), "30..40": (80, 90), "40..50": (90, 100)}
    r = dict(notch=label);
    for k, (a, b_) in sl.items(): r[k] = int(round(diff[a:b_].sum())); r[k + " %"] = round(100 * diff[a:b_].sum() / cf[a:b_].sum(), 1)
    rows.append(r)
D = pd.DataFrame(rows); print("\nD. Count-based differences by slice, window +/-50% (firms, and % of counterfactual in the slice):\n" + D.to_string(index=False)); D.to_csv(f"{OUT}/rev8_count_window50.csv", index=False)

# ---------- E ----------
nxt = pp[["year", "cui", "revenue"]].rename(columns={"revenue": "revenue_next"}); nxt["year"] = nxt.year - 1
pn = pp[["year", "cui", "revenue"]].merge(nxt, on=["year", "cui"], how="inner"); pn["growth"] = pn.revenue_next / pn.revenue - 1
def band_stats(d, lo, hi, thr_next):
    x = d[(d.revenue >= lo) & (d.revenue < hi)]
    return len(x), x.growth.median(), (x.revenue_next > thr_next).mean()
rows = []
for y in (2014, 2018, 2019, 2020, 2021, 2023):                       # ceiling unchanged in y+1
    L = MICRO[y] * fx[y - 1]; L1 = MICRO[y + 1] * fx[y]; q = quantile_of(y, L); q1 = quantile_of(y + 1, L1)
    dy = pn[pn.year == y]
    n1, g1, c1 = band_stats(dy, 0.80 * L, 0.95 * L, L1); n2, g2, c2 = band_stats(dy, 0.65 * L, 0.80 * L, L1); n0, g0, c0 = band_stats(dy, 0.95 * L, L, L1)
    rows.append(dict(year=y, type="notch", n_5_20=n1, growth_5_20=round(100 * g1, 1), cross_5_20=round(100 * c1, 1), growth_20_35=round(100 * g2, 1), growth_0_5=round(100 * g0, 1), diff_5_20_minus_20_35=round(100 * (g1 - g2), 1)))
    for c in controls_for(y, q):
        if c + 1 > 2025 or any(abs(n / value_at(c + 1, q1) - 1) <= 0.25 for n in notches(c + 1)): continue
        Lc = value_at(c, q); Lc1 = value_at(c + 1, q1); dc = pn[pn.year == c]
        n1, g1, c1 = band_stats(dc, 0.80 * Lc, 0.95 * Lc, Lc1); n2, g2, c2 = band_stats(dc, 0.65 * Lc, 0.80 * Lc, Lc1); n0, g0, c0 = band_stats(dc, 0.95 * Lc, Lc, Lc1)
        rows.append(dict(year=y, type=f"control {c}", n_5_20=n1, growth_5_20=round(100 * g1, 1), cross_5_20=round(100 * c1, 1), growth_20_35=round(100 * g2, 1), growth_0_5=round(100 * g0, 1), diff_5_20_minus_20_35=round(100 * (g1 - g2), 1)))
E = pd.DataFrame(rows); print("\nE. Growth t -> t+1 of firms below the notch (median %, share crossing next year's notch %):\n" + E.to_string(index=False)); E.to_csv(f"{OUT}/rev8_growth_below.csv", index=False)
summ = E.groupby(["year", E.type.str.startswith("control").map({True: "controls (mean)", False: "notch"})])[["growth_5_20", "cross_5_20", "growth_20_35", "growth_0_5", "diff_5_20_minus_20_35"]].mean().round(1)
print(summ.to_string()); summ.reset_index().to_csv(f"{OUT}/rev8_growth_below_summary.csv", index=False)

# ---------- F ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q); r = dict(notch=label)
    for half in (0.10, 0.20, 0.30):
        b, m, st, sc, ct, ccs = estimate(y, loc, ctr, half=half); r[f"b_win{int(half*100)}"] = round(b, 2); r[f"excess_win{int(half*100)}"] = int(round((st[int(half/0.01)-5:int(half/0.01)] - sc[int(half/0.01)-5:int(half/0.01)]).sum() * ct.sum()))
    rows.append(r)
Fw = pd.DataFrame(rows); print("\nF. Excess firms in the 5% region by window:\n" + Fw.to_string(index=False)); Fw.to_csv(f"{OUT}/rev8_window_counts.csv", index=False)

# ---------- G ----------
LISTED = {5821, 5829, 6201, 6209, 5510, 5520, 5530, 5590, 5610, 5621, 5629, 5630, 6910, 8621, 8622, 8623, 8690}
d = pp[pp.year == 2025].copy(); d["listed"] = d.caen.fillna(0).astype(int).isin(LISTED)
d["grp"] = np.select([d.micro & ~d.listed, d.micro & d.listed, d.regime == "profit"], ["micro, not listed", "micro, listed activity", "profit tax"], default="other")
edges = np.arange(280_000, 320_001, 2_000); rows = []
for g in ["micro, not listed", "micro, listed activity", "profit tax", "other"]:
    c, _ = np.histogram(d[d.grp == g].revenue, bins=edges); c = c.astype(float)
    ref = np.concatenate([c[4:8], c[10:14]]).mean()             # 288-296k and 300-308k
    rows.append(dict(group=g, firms_280_320k=int(c.sum()), bin_296_298k=int(c[8]), bin_298_300k=int(c[9]), reference_mean_per_bin=round(ref, 1), excess_296_298=round(c[8] / ref - 1, 2), excess_298_300=round(c[9] / ref - 1, 2)))
G = pd.DataFrame(rows); print("\nG. 2025 revenue 296-300k lei by regime and listed activity (2,000-lei bins; reference = mean of 288-296k and 300-308k):\n" + G.to_string(index=False)); G.to_csv(f"{OUT}/rev8_60k_by_regime.csv", index=False)

# ---------- H ----------
L = 100e3 * fx[2015]; rows = []
for y in (2014, 2015, 2016):
    e = pp[pp.year == y]; r15 = rev[2015]; q = np.searchsorted(r15, L) / len(r15); Lc = value_at(y, q) if y != 2015 else L
    for lab_, lo, hi in [("5-0% below", 0.95, 1.0), ("0-3% above", 1.0, 1.03), ("3-10% above", 1.03, 1.10), ("10-20% above", 1.10, 1.20)]:
        x = e[(e.revenue >= lo * Lc) & (e.revenue < hi * Lc)]
        rows.append(dict(year=y, band=lab_, location=round(Lc), n=len(x), median_margin=round(x.margin.median(), 3), share_margin_lt_6_25=round(100 * (x.margin < 0.0625).mean(), 1), share_margin_lt_18_75=round(100 * (x.margin < 0.1875).mean(), 1), micro_share=round(100 * x.micro.mean(), 1)))
H = pd.DataFrame(rows); print("\nH. 2015 year-end notch (EUR 100k at end-2015 rate = {:,.0f} lei): margins by position, 2015 vs same quantile in 2014/2016:\n".format(L) + H.to_string(index=False)); H.to_csv(f"{OUT}/rev8_2015_yearend_margins.csv", index=False)
edges = np.arange(430_000, 480_001, 2_000)
Hh = pd.DataFrame({"bin_lo": edges[:-1], **{str(y): np.histogram(pp[pp.year == y].revenue, bins=edges)[0] for y in (2014, 2015, 2016)}}); Hh.to_csv(f"{OUT}/rev8_2015_yearend_hist.csv", index=False)
print(Hh.T.to_string())

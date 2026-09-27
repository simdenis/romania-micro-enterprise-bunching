"""Ninth round.
A. Fiscal table for 2023 rebuilt on the ANAF registry flag (31 Aug 2023) instead of the accounts inference.
B. The 2015 escape test: firms 0-3% above the EUR 100k-for-2016 ceiling on 2015 revenue vs 5-0% below: regime in 2016,
   tax pattern in 2016, revenue change 2015->2016; placebo cohorts at the same quantile in 2014->2015 and 2017->2018.
C. Count-based excess in the 5 bins below the notch for windows of 10/20/30% (should be identical by construction).
D. Narrow-band DiD on MEAN margins (winsorised at +/-1) with contemporaneous placebos, bootstrap SE; per-switch on means
   for the entry design.
E. DiB within three lagged-margin groups: gain (t-1 margin > break-even), low positive (0 < margin <= break-even), loss (<= 0).
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pd.set_option("display.width", 260)
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax", "tax"]).drop_duplicates(["year", "cui"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")[["year", "cui", "regime"]]
pp = pp.merge(reg, on=["year", "cui"], how="left"); pp["micro"] = pp.regime.isin(["micro", "micro_loss"]); pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1)
pp = pp[pp.revenue > 0].copy(); N = {y: len(rev[y]) for y in rev}
def controls_for(y, q): return [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
def anaf_flags(path):
    a = pd.read_csv(path, sep="|", encoding="utf-8-sig", usecols=["COD_FISCAL", "IMP100", "IMP120"], dtype=str, on_bad_lines="skip", engine="c", quoting=3, index_col=False)
    a["cui"] = pd.to_numeric(a.COD_FISCAL, errors="coerce"); a = a.dropna(subset=["cui"]).drop_duplicates("cui"); a["cui"] = a.cui.astype("int64")
    m = (a.IMP120.str.strip() == "DA"); p = (a.IMP100.str.strip() == "DA")
    a["flag"] = np.select([m & ~p, p & ~m, m & p], ["micro", "profit", "both"], default="none"); return a.set_index("cui")["flag"]

# ---------- A ----------
F23 = anaf_flags(f"{ROOT}/data/raw/anaf/date_identificare_platitori_2023.csv")
d = pp[pp.year == 2023].copy(); d["flag"] = d.cui.map(F23).fillna("not in registry"); d["rev_eur_prev"] = d.revenue / fx[2022]
bands = [(0, 60e3, "< 60k"), (60e3, 250e3, "60-250k"), (250e3, 500e3, "250-500k"), (500e3, 1e6, "500k-1M"), (1e6, 1e12, "> 1M")]
rows = []
for lo, hi, lab in bands:
    x = d[(d.rev_eur_prev >= lo) & (d.rev_eur_prev < hi)]; mic = x[x.flag == "micro"]; prof = x[x.flag == "profit"]
    rows.append(dict(band_eur=lab, firms=len(x), micro_firms=len(mic), micro_tax_paid_mlei=round(mic.tax.sum() / 1e6, 1), micro_tax_over_rev_pct=round(100 * mic.tax.sum() / mic.revenue.sum(), 2),
                     profit_tax_if_16pct_mlei=round(0.16 * mic.pretax.clip(lower=0).sum() / 1e6, 1), micro_pretax_mlei=round(mic.pretax.sum() / 1e6, 1),
                     profit_firms=len(prof), profit_tax_paid_mlei=round(prof.tax.sum() / 1e6, 1), inferred_micro_firms=int(x.micro.sum()), inferred_micro_tax_mlei=round(x[x.micro].tax.sum() / 1e6, 1)))
A = pd.DataFrame(rows); A["gap_mlei"] = (A.profit_tax_if_16pct_mlei - A.micro_tax_paid_mlei).round(1); A["gap_share"] = (100 * A.gap_mlei / A.gap_mlei.sum()).round(1)
print("A. 2023 fiscal table on the ANAF flag (million lei):\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev9_fiscal_anaf.csv", index=False)
print(f"   totals: micro tax {A.micro_tax_paid_mlei.sum():,.0f}; 16% of profit {A.profit_tax_if_16pct_mlei.sum():,.0f}; gap {A.gap_mlei.sum():,.0f}; share below 500k {A.gap_share[:3].sum():.1f}; 500k-1M {A.gap_share.iloc[3]:.1f}")
print("   crosstab flag x inferred regime, > 1M band:"); xx = d[d.rev_eur_prev >= 1e6]; print(pd.crosstab(xx.flag, xx.regime))

# ---------- B ----------
nxt = pp[["year", "cui", "revenue", "pretax", "tax", "regime", "micro"]].rename(columns={"revenue": "rev1", "pretax": "pretax1", "tax": "tax1", "regime": "regime1", "micro": "micro1"}); nxt["year"] = nxt.year - 1
pn = pp.merge(nxt, on=["year", "cui"], how="left")
L = 100e3 * fx[2015]; q = quantile_of(2015, L); rows = []
for y in (2014, 2015, 2017):
    Ly = L if y == 2015 else value_at(y, q); dy = pn[pn.year == y]
    for lab, lo, hi in [("5-0% below", 0.95, 1.0), ("0-3% above", 1.0, 1.03), ("3-10% above", 1.03, 1.10)]:
        x = dy[(dy.revenue >= lo * Ly) & (dy.revenue < hi * Ly)]; xs = x.dropna(subset=["rev1"]); xs = xs[xs.rev1 > 0]
        pr = xs[xs.regime1 == "profit"]; mi = xs[xs.micro1]
        rows.append(dict(cohort=f"{y}->{y+1}", band=lab, location=round(Ly), n=len(x), n_next=len(xs), exit_pct=round(100 * (1 - len(xs) / len(x)), 1),
                         next_micro_pct=round(100 * xs.micro1.mean(), 1), next_profit_pct=round(100 * (xs.regime1 == "profit").mean(), 1), next_notax_pct=round(100 * (xs.regime1 == "no_tax").mean(), 1),
                         next_tax_over_rev_pct=round(100 * (xs.tax1 / xs.rev1).median(), 2), next_tax_over_pretax_pct_profitpattern=round(100 * (pr.tax1 / pr.pretax1).median(), 1) if len(pr) else np.nan,
                         this_margin=round(x.margin.median(), 3), next_margin=round((xs.pretax1 / xs.rev1).clip(-1, 1).median(), 3),
                         next_rev_growth_pct=round(100 * (xs.rev1 / xs.revenue - 1).median(), 1), next_below_location_pct=round(100 * (xs.rev1 < Ly).mean(), 1), next_below_97pct=round(100 * (xs.rev1 < 0.97 * Ly).mean(), 1)))
B = pd.DataFrame(rows); print("\nB. 2015 escape test (next-year regime and revenue by band; 2014 and 2017 cohorts at the same quantile as placebos):\n" + B.to_string(index=False)); B.to_csv(f"{OUT}/rev9_escape_2015.csv", index=False)

# ---------- C ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q); r = dict(notch=label)
    for half in (0.10, 0.20, 0.30):
        k0 = int(round(half / 0.01)); st = rel_hist(y, loc, 0.01, half) / N[y]; sc = np.mean([rel_hist(c, value_at(c, q), 0.01, half) / N[c] for c in ctr], axis=0)
        r[f"count_excess_win{int(half*100)}"] = int(round((st[k0 - 5:k0] - sc[k0 - 5:k0]).sum() * N[y]))
    rows.append(r)
C = pd.DataFrame(rows); print("\nC. Count-based excess (5 bins below) by window:\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/rev9_count_window.csv", index=False)

# ---------- D ----------
pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y]); rng2 = np.random.default_rng(21)
def build(base, years, cut, band=0.20, lo_c=None, hi_c=None):
    d = pp[pp.year.isin(years)]; cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index)
    a = pp[pp.year.isin([base - 1, base]) & pp.cui.isin(keep)].groupby("cui").rev_eur.mean()
    lo_c = 1.0 if lo_c is None else lo_c; hi_c = 1 + band if hi_c is None else hi_c
    T = set(a[(a > (1 - band) * cut) & (a <= cut)].index); Cs = set(a[(a > lo_c * cut) & (a <= hi_c * cut)].index)
    w = d[d.cui.isin(T | Cs)].pivot(index="cui", columns="year", values="margin"); w["T"] = w.index.isin(T); return w
def did_mean(w, base, y): return (w.loc[w["T"], y].mean() - w.loc[w["T"], base].mean()) - (w.loc[~w["T"], y].mean() - w.loc[~w["T"], base].mean())
fs = pd.read_csv(f"{OUT}/rev7_did_firststage.csv"); rows = []
for lab, base, years in (("2018 entry", 2017, list(range(2015, 2021))), ("2023 exit", 2022, list(range(2020, 2026)))):
    W = build(base, years, 1e6); P15 = build(base, years, 1.5e6); P20 = build(base, years, 2e6); W2 = build(base, years, 1e6, lo_c=1.2, hi_c=1.5)
    for y in years:
        raw, p15, p20, raw2 = did_mean(W, base, y), did_mean(P15, base, y), did_mean(P20, base, y), did_mean(W2, base, y); corr = raw - 0.5 * (p15 + p20)
        bs_raw, bs_corr = [], []
        if y != base:
            for _ in range(200):
                s = lambda X: X.sample(len(X), replace=True, random_state=rng2.integers(1e9))
                r_ = did_mean(s(W), base, y); c_ = r_ - 0.5 * (did_mean(s(P15), base, y) + did_mean(s(P20), base, y)); bs_raw.append(r_); bs_corr.append(c_)
        f1 = fs[(fs.design == lab) & (fs.control == "control (1.0,1.2]") & (fs.year == y)].first_stage_pp; f2 = fs[(fs.design == lab) & (fs.control == "control (1.2,1.5]") & (fs.year == y)].first_stage_pp
        f1 = float(f1.iloc[0]) if len(f1) else np.nan; f2 = float(f2.iloc[0]) if len(f2) else np.nan
        rows.append(dict(event=lab, year=y, horizon=y - base, base_mean_T=round(100 * W.loc[W["T"], base].mean(), 2), raw_mean_DiD_pp=round(100 * raw, 2), raw_se_pp=round(100 * np.std(bs_raw), 2) if bs_raw else np.nan,
                         placebo_1_5M_pp=round(100 * p15, 2), placebo_2M_pp=round(100 * p20, 2), corrected_pp=round(100 * corr, 2), corrected_se_pp=round(100 * np.std(bs_corr), 2) if bs_corr else np.nan,
                         wide_control_DiD_pp=round(100 * raw2, 2), first_stage_pp=f1, per_switch_pp=round(1e4 * raw / f1, 2) if (lab == "2018 entry" and f1 == f1 and abs(f1) > 5) else np.nan,
                         per_switch_corrected_pp=round(1e4 * corr / f1, 2) if (lab == "2018 entry" and f1 == f1 and abs(f1) > 5) else np.nan,
                         per_switch_wide_pp=round(1e4 * raw2 / f2, 2) if (lab == "2018 entry" and f2 == f2 and abs(f2) > 5) else np.nan))
D = pd.DataFrame(rows); print("\nD. Narrow-band DiD on mean margins (winsorised at +/-1), contemporaneous placebos, per-switch on means (entry):\n" + D.to_string(index=False)); D.to_csv(f"{OUT}/rev9_event_study_means.csv", index=False)

# ---------- E ----------
lag = pp[["year", "cui", "margin"]].rename(columns={"margin": "margin_lag"}); lag["year"] = lag.year + 1
pl = pp.merge(lag, on=["year", "cui"], how="inner")
CUT = {y: (0.1875 if y in (2014, 2015, 2024, 2025) else 0.0625) for y in range(2014, 2026)}
def hist_arr(v, loc, bw=0.01, half=0.20):
    edges = loc * (1 + np.arange(-half, half + bw / 2, bw)); c, _ = np.histogram(v, bins=edges); return c.astype(float)
rows = []
for label, y, loc in [(l, y, loc) for l, y, loc in TARGETS if "micro" in l and y >= 2015]:
    q = quantile_of(y, loc); ctr = [c for c in controls_for(y, q) if c >= 2015]; cut = CUT[y]
    for g, cond in (("gain", lambda d: d.margin_lag > cut), ("low positive", lambda d: (d.margin_lag > 0) & (d.margin_lag <= cut)), ("loss", lambda d: d.margin_lag <= 0)):
        arr = {c: np.sort(pl[(pl.year == c) & cond(pl)].revenue.values) for c in [y] + ctr}
        qg = np.searchsorted(arr[y], loc) / len(arr[y]); ct = hist_arr(arr[y], loc); st = ct / ct.sum()
        ccs = [hist_arr(arr[c], arr[c][min(int(qg * len(arr[c])), len(arr[c]) - 1)]) for c in ctr]; sc = np.mean([cc / cc.sum() for cc in ccs], axis=0)
        b = (st[15:20] - sc[15:20]).sum() / sc[15:20].mean(); exc = (st[15:20] - sc[15:20]).sum() * ct.sum(); se = bootstrap(ct, ccs, 0.01, 0.05, 0.20, reps=200)
        rows.append(dict(notch=label, cut_pct=round(100 * cut, 2), group=g, firms_in_window=int(ct.sum()), b=round(b, 2), se=round(se, 2), excess_firms=int(round(exc)), excess_pct_of_window=round(100 * exc / ct.sum(), 1), group_share_of_year=round(100 * len(arr[y]) / len(pl[pl.year == y]), 1)))
E = pd.DataFrame(rows); print("\nE. DiB within three lagged-margin groups:\n" + E.to_string(index=False)); E.to_csv(f"{OUT}/rev9_dib_by_margin3.csv", index=False)

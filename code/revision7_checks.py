"""Seventh round (spine review).
A. Count-based DiB: bins as shares of ALL positive-revenue firms in the year (count/N_t), not of the window, so the
   integration constraint becomes a test: excess below vs deficit above (5 bins and full 20%) in firms.
B. The 3% puzzle: share of bunchers / just-above with margin > 18.75% (and > 6.25%), and the rate the median buncher paid.
C. Margin DiD first stage: micro share by year for the narrow bands; sensitivity with a (1.2, 1.5] x cut-off control band;
   effect per regime switch = DiD / (change in micro share T - change in micro share C).
D. Placebo distribution: percentiles of |b| and the location of the maximum.
E. 2025 year-end notch: 1,000-lei histogram 480-520k lei, mass below vs above 497,410 and at 499-500k.
F. Fiscal magnitudes, 2023: micro tax paid vs 16% of reported pre-tax profit, by revenue band; and the static-simulation
   correction implied by the margin response for reclassified firms.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax", "tax", "employees"]).drop_duplicates(["year", "cui"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")[["year", "cui", "regime"]]
pp = pp.merge(reg, on=["year", "cui"], how="left"); pp["micro"] = pp.regime.isin(["micro", "micro_loss"]); pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1)
N = {y: len(rev[y]) for y in rev}
def controls_for(y, q): return [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]

# ---------- A ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q)
    ct = rel_hist(y, loc, 0.01, 0.20); st = ct / N[y]
    sc = np.mean([rel_hist(c, value_at(c, q), 0.01, 0.20) / N[c] for c in ctr], axis=0)
    diff = (st - sc) * N[y]                                   # firms, relative to counterfactual scaled to year t
    exc5 = diff[15:20].sum(); def5 = diff[20:25].sum(); def20 = diff[20:40].sum(); below_far = diff[0:15].sum()
    b_cnt = (st[15:20] - sc[15:20]).sum() / sc[15:20].mean()
    rows.append(dict(notch=label, N_t=N[y], b_count=round(b_cnt, 2), excess_5below=int(round(exc5)), deficit_5above=int(round(def5)), deficit_20above=int(round(def20)),
                     diff_5to20below=int(round(below_far)), net_window=int(round(diff.sum())), share_of_excess_found_above=round(-def20 / exc5, 2) if exc5 > 0 else np.nan))
A = pd.DataFrame(rows); pd.set_option("display.width", 260); print("A. Count-based (share of all firms) DiB:\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev7_count_based.csv", index=False)

# ---------- B ----------
rows = []
for y in range(2014, 2026):
    L = MICRO[y] * fx[y - 1]; d = pp[(pp.year == y) & (pp.revenue > 0)].copy(); d["rel"] = d.revenue / L - 1
    for g, m in [("bunchers", (d.rel >= -0.05) & (d.rel < 0)), ("just_above", (d.rel >= 0) & (d.rel < 0.10))]:
        x = d[m]; rate = (x.tax / x.revenue)[x.micro].median()
        rows.append(dict(year=y, group=g, n=len(x), median_margin=round(x.margin.median(), 3), share_margin_gt_6_25=round(100 * (x.margin > 0.0625).mean(), 1),
                         share_margin_gt_18_75=round(100 * (x.margin > 0.1875).mean(), 1), median_rate_paid_by_micro=round(100 * rate, 2) if rate == rate else np.nan, micro_share=round(100 * x.micro.mean(), 1)))
B = pd.DataFrame(rows); print("\nB. Margins and rates by position:\n" + B.to_string(index=False)); B.to_csv(f"{OUT}/rev7_margins_3pct.csv", index=False)

# ---------- C ----------
pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y])
def build(base, years, cut, lo_t=0.8, hi_t=1.0, lo_c=1.0, hi_c=1.2):
    d = pp[pp.year.isin(years) & (pp.revenue > 0)]; cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index)
    a = pp[pp.year.isin([base - 1, base]) & pp.cui.isin(keep)].groupby("cui").rev_eur.mean()
    T = set(a[(a > lo_t * cut) & (a <= hi_t * cut)].index); C = set(a[(a > lo_c * cut) & (a <= hi_c * cut)].index)
    w = d[d.cui.isin(T | C)].pivot(index="cui", columns="year", values="margin"); ms = d[d.cui.isin(T | C)].pivot(index="cui", columns="year", values="micro")
    w["T"] = w.index.isin(T); ms["T"] = ms.index.isin(T); return w, ms
def did(w, base, y): return (w.loc[w["T"], y].median() - w.loc[w["T"], base].median()) - (w.loc[~w["T"], y].median() - w.loc[~w["T"], base].median())
rows = []
for lab, base, years in (("2018 entry", 2017, list(range(2015, 2021))), ("2023 exit", 2022, list(range(2020, 2026)))):
    for ctrl_lab, lo_c, hi_c in (("control (1.0,1.2]", 1.0, 1.2), ("control (1.2,1.5]", 1.2, 1.5)):
        w, ms = build(base, years, 1e6, lo_c=lo_c, hi_c=hi_c)
        for y in years:
            mT, mC = ms.loc[ms["T"], y].mean(), ms.loc[~ms["T"], y].mean(); dT, dC = mT - ms.loc[ms["T"], base].mean(), mC - ms.loc[~ms["T"], base].mean()
            dd = did(w, base, y); fs = dT - dC
            rows.append(dict(design=lab, control=ctrl_lab, year=y, micro_T=round(100 * mT, 1), micro_C=round(100 * mC, 1), first_stage_pp=round(100 * fs, 1), DiD_pp=round(100 * dd, 2),
                             per_switch_pp=round(100 * dd / fs, 2) if abs(fs) > 0.05 else np.nan, nT=int(ms["T"].sum()), nC=int((~ms["T"]).sum())))
C = pd.DataFrame(rows); print("\nC. First stage (micro shares) and per-switch effects:\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/rev7_did_firststage.csv", index=False)

# ---------- D ----------
pl = pd.read_csv(f"{OUT}/rev_placebo_locations.csv"); ab = pl.b.abs()
print(f"\nD. placebo |b|: n={len(pl)}, sd(b)={pl.b.std():.3f}, p90={ab.quantile(.9):.2f}, p95={ab.quantile(.95):.2f}, p99={ab.quantile(.99):.2f}, max={ab.max():.2f} at year {int(pl.loc[ab.idxmax(),'year'])}, mult {pl.loc[ab.idxmax(),'mult']}, location {int(pl.loc[ab.idxmax(),'location_lei']):,} lei")
print("   top 5 |b| placebos:"); print(pl.reindex(ab.sort_values(ascending=False).index).head(5)[["year", "mult", "location_lei", "n_controls", "b", "m"]].to_string(index=False))

# ---------- E ----------
v = rev[2025]; loc = 100e3 * fx[2024]; edges = np.arange(470_000, 530_001, 1_000); c, _ = np.histogram(v, bins=edges)
med = np.median(c); print(f"\nE. 2025 revenue, 1,000-lei bins, 470-530k: median bin {med:.0f}")
for a in range(490_000, 505_000, 1_000): print(f"   {a:,}-{a+1000:,}: {c[(a-470_000)//1000]}")
print(f"   firms 492-497.41k: {c[22:27].sum()}, 497.41-500k: {c[27:30].sum()}, 500-505k: {c[30:35].sum()}; location {loc:,.0f}")

# ---------- F ----------
y = 2023; d = pp[(pp.year == y) & (pp.revenue > 0)].copy(); d["rev_eur_prev"] = d.revenue / fx[y - 1]
bands = [(0, 60e3, "< 60k"), (60e3, 250e3, "60-250k"), (250e3, 500e3, "250-500k"), (500e3, 1e6, "500k-1M"), (1e6, 1e12, "> 1M")]
rows = []
for lo, hi, lab in bands:
    x = d[(d.rev_eur_prev >= lo) & (d.rev_eur_prev < hi)]; mic = x[x.micro]; prof = x[x.regime == "profit"]
    rows.append(dict(band_eur=lab, firms=len(x), micro_firms=len(mic), micro_tax_paid_mlei=round(mic.tax.sum() / 1e6, 1), profit_tax_if_16pct_mlei=round(0.16 * mic.pretax.clip(lower=0).sum() / 1e6, 1),
                     micro_pretax_mlei=round(mic.pretax.sum() / 1e6, 1), profit_firms=len(prof), profit_tax_paid_mlei=round(prof.tax.sum() / 1e6, 1)))
F = pd.DataFrame(rows); F["gap_mlei"] = (F.profit_tax_if_16pct_mlei - F.micro_tax_paid_mlei).round(1); F["gap_share"] = (100 * F.gap_mlei / F.gap_mlei.sum()).round(1)
print("\nF. 2023, firms by revenue band (previous year-end rate): micro tax paid vs 16% of reported pre-tax profit (million lei):\n" + F.to_string(index=False)); F.to_csv(f"{OUT}/rev7_fiscal.csv", index=False)
# static-simulation correction for reclassified firms: margin falls by DiD relative to base margin
ev = pd.read_csv(f"{OUT}/rev6_event_study_contemp.csv"); ex = ev[(ev.event == "2023 exit") & (ev.horizon > 0)]
w23, _ = build(2022, list(range(2020, 2026)), 1e6); base_margin = float(w23.loc[w23["T"], 2022].median()); w18, _ = build(2017, list(range(2015, 2021)), 1e6); base18 = float(w18.loc[w18["T"], 2017].median())
print(f"   narrow-band base margins: 2022 treated {100*base_margin:.2f}%, 2017 treated {100*base18:.2f}%")
print(f"   2023 exit DiD (pp): {ex.raw_DiD_pp.tolist()} -> relative to a base margin of {100*base_margin:.1f}%: {[round(-v/(100*base_margin)*100) for v in ex.raw_DiD_pp]} % lower reported profit, i.e. static profit-tax revenue from reclassified firms overstated by that share")

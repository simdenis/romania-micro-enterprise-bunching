"""Revision checks answering the referee reports.
A. Placebo distribution of the DiB statistic at non-notch locations (uncertainty beyond the bootstrap).
B. 2025 year-end notch: EUR 100k at the end-2024 rate (497,410 lei) vs end-2025 rate (509,850) vs round 500,000 lei.
C. 2023 transition claim with a 2021 placebo (inflow from above the notch location is mean reversion?).
D. Profit margins of bunchers vs firms just above (the notch is a tax increase only for high-margin firms).
E. Age-standardised splitting comparison (bunchers are younger; the 18-month window is age-related).
F. Regime cross-check on the full sample.
Outputs to outputs/rev_*.csv."""
import os, itertools, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]
exec(compile(src, "dib", "exec"))                      # loads p, rev, fx, MICRO, notches(), quantile_of, value_at, estimate, rel_hist
rng = np.random.default_rng(3)

# ---------- A. placebo locations ----------
rows = []
for y in range(2014, 2026):
    for mult in (0.45, 0.6, 0.75, 1.35, 1.6, 2.0, 2.5):
        loc = MICRO[y] * fx[y - 1] * mult
        if any(abs(n / loc - 1) <= 0.25 for n in notches(y)): continue          # location must itself be notch-free
        q = quantile_of(y, loc)
        controls = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
        if len(controls) < 3: continue
        b, m, st, sc, ct, ccs = estimate(y, loc, controls)
        rows.append(dict(year=y, mult=mult, location_lei=round(loc), n_controls=len(controls), firms=int(ct.sum()), b=round(b, 3), m=round(m, 3)))
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/rev_placebo_locations.csv", index=False)
print(f"A. placebo DiB at {len(A)} non-notch locations: mean b {A.b.mean():.3f}, sd {A.b.std():.3f}, max |b| {A.b.abs().max():.2f}; mean m {A.m.mean():.3f}, sd {A.m.std():.3f}")

# ---------- B. 2025 year-end notch location ----------
rows = []
for label, loc in [("EUR100k x end-2024 rate", 100e3 * fx[2024]), ("EUR100k x end-2025 rate", 100e3 * fx[2025]), ("500,000 lei round number", 500_000.0)]:
    q = quantile_of(2025, loc); controls = [c for c in range(2014, 2025) if all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
    b, m, st, sc, ct, ccs = estimate(2025, loc, controls, bw=0.005, region=0.02)
    rows.append(dict(candidate=label, location_lei=round(loc), b_2pct_region=round(b, 2), excess_firms=int(round((st[36:40] - sc[36:40]).sum() * ct.sum()))))
v = rev[2025]; bins = np.arange(480_000, 520_001, 1_000); c1k, _ = np.histogram(v, bins=bins)
peak = bins[np.argmax(c1k)]
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/rev_2025_yearend_location.csv", index=False)
print(f"\nB. 2025 year-end notch candidates (2% region, 0.5% bins):\n{B.to_string(index=False)}\n   1,000-lei modal bin in 480-520k: {peak:,}-{peak+1000:,} lei ({c1k.max()} firms); 100k x end-2024 = {100e3*fx[2024]:,.0f}")

# ---------- C. 2023 transition with 2021 placebo ----------
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax", "employees"]).drop_duplicates(["year", "cui"])
def origin(y, loc_prev_rate):
    cur = pp[pp.year == y].set_index("cui").revenue / fx[y - 1]; prev = pp[pp.year == y - 1].set_index("cui").revenue / fx[y - 2]
    L = loc_prev_rate
    def share_above(ids): x = prev.reindex(ids); return round(100 * (x > L).mean(), 1), round(100 * ((x > 1.2 * L) & (x <= 2 * L)).mean(), 1)
    bunch = cur[(cur >= 0.94 * L) & (cur < L)].index; ctrl = cur[(cur >= 0.80 * L) & (cur < 0.86 * L)].index
    return dict(year=y, location_eur=L, bunchers_n=len(bunch), bunchers_from_above=share_above(bunch)[0], bunchers_from_120_200pct=share_above(bunch)[1],
                controls_n=len(ctrl), controls_from_above=share_above(ctrl)[0], controls_from_120_200pct=share_above(ctrl)[1])
C = pd.DataFrame([origin(2023, 500e3), origin(2024, 500e3), origin(2021, 500e3), origin(2020, 500e3), origin(2019, 500e3)])
C["ratio"] = (C.bunchers_from_above / C.controls_from_above).round(2); C.to_csv(f"{OUT}/rev_transition_placebo.csv", index=False)
print("\nC. Share of firms just below EUR 500k in year t that were above EUR 500k in t-1 (2023 = notch year; others placebo):\n" + C.to_string(index=False))

# ---------- D. margins ----------
rows = []
for y in range(2018, 2026):
    d = pp[(pp.year == y) & (pp.revenue > 0)].copy(); L = MICRO[y] * fx[y - 1]; d["rel"] = d.revenue / L - 1; d["margin"] = d.pretax / d.revenue
    for g, msk in [("bunchers 0-5% below", (d.rel >= -0.05) & (d.rel < 0)), ("just above 0-10%", (d.rel >= 0) & (d.rel < 0.10)), ("10-25% below", (d.rel >= -0.25) & (d.rel < -0.10))]:
        x = d.loc[msk, "margin"]; rows.append(dict(year=y, group=g, firms=int(msk.sum()), median_margin=round(x.median(), 3), share_margin_above_6pct=round(100 * (x > 0.0625).mean(), 1)))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/rev_margins.csv", index=False)
print("\nD. Pre-tax profit margin by position (median):\n" + D.pivot(index="year", columns="group", values="median_margin").to_string())

# ---------- E. age-standardised splitting ----------
firms = pd.read_parquet(f"{ROOT}/data/panel/onrc_firms.parquet", columns=["cui", "reg_date"]); age = firms.set_index("cui").reg_date
A2 = pd.read_csv(f"{OUT}/splitting_by_position.csv")     # group shares only; need firm-level flags -> recompute quickly from splitting_test internals
exec(open(f"{ROOT}/scripts/splitting_test.py").read().split("rows = []")[0])     # loads p (SRL panel), sib, asib, flag_window
rows = []
for y in range(2018, 2024):
    d = p[p.year == y].copy(); d["sib_new"] = d.cui.isin(flag_window(sib, y)); d["addr_new"] = d.cui.isin(flag_window(asib, y))
    d["age"] = y - age.reindex(d.cui).dt.year.values; d["band"] = pd.cut(d.age, [-1, 2, 5, 10, 100], labels=["0-2", "3-5", "6-10", "11+"])
    d["grp"] = np.select([(d.rel >= -0.05) & (d.rel < 0), (d.rel >= 0) & (d.rel < 0.10)], ["bunchers", "just_above"], default="other")
    rows.append(d[d.grp != "other"][["year", "grp", "band", "sib_new", "addr_new"]])
E = pd.concat(rows)
byband = E.groupby(["grp", "band"], observed=True)[["sib_new", "addr_new"]].mean() * 100; n = E.groupby(["grp", "band"], observed=True).size()
wts = E[E.grp == "bunchers"].band.value_counts(normalize=True)          # standardise to bunchers' age distribution
std = {g: {m: sum(byband.loc[(g, b), m] * wts[b] for b in wts.index) for m in ["sib_new", "addr_new"]} for g in ["bunchers", "just_above"]}
tab = byband.unstack(0); tab.columns = [f"{m}_{g}" for m, g in tab.columns]; tab["n_bunchers"] = n.unstack(0)["bunchers"]; tab["n_just_above"] = n.unstack(0)["just_above"]
tab.to_csv(f"{OUT}/rev_splitting_by_age.csv")
print("\nE. Sibling measures by age band, pooled 2018-2023 (%):\n" + tab.round(2).to_string())
print(f"   age-standardised (bunchers' age mix): admin-new bunchers {std['bunchers']['sib_new']:.2f} vs just above {std['just_above']['sib_new']:.2f} "
      f"(+{100*(std['bunchers']['sib_new']/std['just_above']['sib_new']-1):.0f}%); address-new {std['bunchers']['addr_new']:.2f} vs {std['just_above']['addr_new']:.2f} "
      f"(+{100*(std['bunchers']['addr_new']/std['just_above']['addr_new']-1):.0f}%)")
print("   bunchers age mix:", wts.round(3).to_dict(), " just-above age mix:", E[E.grp == "just_above"].band.value_counts(normalize=True).round(3).to_dict())

# ---------- F. cross-check, full sample ----------
ct = pd.read_csv(f"{OUT}/regime_crosscheck_2023.csv", index_col=0)
tot = ct.values.sum(); matched = tot - ct["not_in_anaf"].sum()
micro_flag = ct["flag_micro"].sum(); prof_flag = ct["flag_profit"].sum()
print(f"\nF. 2023 cross-check, all {tot:,} firms: matched {100*matched/tot:.1f}%. Among ANAF micro-flagged ({micro_flag:,}): inferred micro/micro_loss {100*(ct.loc['micro','flag_micro']+ct.loc['micro_loss','flag_micro'])/micro_flag:.1f}%, no_tax {100*ct.loc['no_tax','flag_micro']/micro_flag:.1f}%, profit {100*ct.loc['profit','flag_micro']/micro_flag:.1f}%. "
      f"Among ANAF profit-flagged ({prof_flag:,}): inferred profit {100*ct.loc['profit','flag_profit']/prof_flag:.1f}%, no_tax {100*ct.loc['no_tax','flag_profit']/prof_flag:.1f}%, micro/micro_loss {100*(ct.loc['micro','flag_profit']+ct.loc['micro_loss','flag_profit'])/prof_flag:.1f}%.")

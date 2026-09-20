"""Fifth round.
A. Margin event study, narrowed: bands within +/-20% of the EUR 1M cut-off, assigned on the two-year average of revenue
   (base year and the year before), balanced panels; placebo-corrected DiD = event DiD minus placebo DiD at the same
   horizon (placebo bands assigned on 2018-2019 average, base 2019); SE by bootstrap over firms (200 draws).
B. Matched locations L*_c and distance to the nearest notch for every control year (transparency table).
C. The EUR 60k rate boundary in 2025, when VAT moved to 395k lei: histogram of 2025 revenue 250-450k lei, and
   polynomial excess at 60k x e_2024 = 298,446 lei, with 2024 (confounded) and 2023 (no boundary) for comparison.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax", "turnover"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")[["year", "cui", "regime"]]
pp = pp.merge(reg, on=["year", "cui"], how="left"); pp["micro"] = pp.regime.isin(["micro", "micro_loss"])
pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1); pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y])
rng = np.random.default_rng(5)

# ---------- A ----------
def build(base, years, cut=1e6, band=0.20):
    d = pp[pp.year.isin(years) & (pp.revenue > 0)]
    cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index)
    a = pp[pp.year.isin([base - 1, base]) & pp.cui.isin(keep)].groupby("cui").rev_eur.mean()      # two-year average
    T = set(a[(a > (1 - band) * cut) & (a <= cut)].index); C = set(a[(a > cut) & (a <= (1 + band) * cut)].index)
    w = d[d.cui.isin(T | C)].pivot(index="cui", columns="year", values="margin")
    w["T"] = w.index.isin(T); return w
def did(w, base, years, stat="median"):
    f = (lambda s: s.median()) if stat == "median" else (lambda s: s.mean())
    out = {}
    for y in years:
        out[y] = (f(w.loc[w["T"], y]) - f(w.loc[w["T"], base])) - (f(w.loc[~w["T"], y]) - f(w.loc[~w["T"], base]))
    return out
Y18, Y23, Y19 = list(range(2015, 2021)), list(range(2020, 2026)), list(range(2017, 2022))
W18, W23, W19 = build(2017, Y18), build(2022, Y23), build(2019, Y19)
print(f"A. narrowed bands (two-year average revenue within 20% of EUR 1M): 2018 event T={W18['T'].sum():,} C={(~W18['T']).sum():,}; 2023 event T={W23['T'].sum():,} C={(~W23['T']).sum():,}; placebo T={W19['T'].sum():,} C={(~W19['T']).sum():,}")
rows = []
for stat in ("median", "mean"):
    d18, d23, d19 = did(W18, 2017, Y18, stat), did(W23, 2022, Y23, stat), did(W19, 2019, Y19, stat)
    # horizons: h = year - base; placebo at same horizon
    for lab, dd, base in (("2018 entry", d18, 2017), ("2023 exit", d23, 2022), ("placebo 2019", d19, 2019)):
        for y, v in dd.items():
            h = y - base; pv = d19.get(2019 + h, np.nan)
            rows.append(dict(stat=stat, event=lab, year=y, horizon=h, DiD_pp=round(100 * v, 2), placebo_same_horizon_pp=round(100 * pv, 2) if pv == pv else np.nan, corrected_pp=round(100 * (v - pv), 2) if (pv == pv and lab != "placebo 2019") else np.nan))
    # bootstrap SE for corrected estimates at h=+1,+2 (median stat only to keep runtime)
    if stat == "median":
        for lab, W, base, ys in (("2018 entry", W18, 2017, Y18), ("2023 exit", W23, 2022, Y23)):
            for h in (1, 2, 3):
                if base + h not in ys or 2019 + h not in Y19: continue
                bs = []
                for _ in range(200):
                    Wb = W.sample(len(W), replace=True, random_state=rng.integers(1e9)); Pb = W19.sample(len(W19), replace=True, random_state=rng.integers(1e9))
                    bs.append(did(Wb, base, [base + h])[base + h] - did(Pb, 2019, [2019 + h])[2019 + h])
                r = [x for x in rows if x["stat"] == "median" and x["event"] == lab and x["horizon"] == h][0]; r["corrected_se_pp"] = round(100 * np.std(bs), 2)
A = pd.DataFrame(rows); pd.set_option("display.width", 250); print(A[A.stat == "median"].to_string(index=False)); print(A[A.stat == "mean"].to_string(index=False)); A.to_csv(f"{OUT}/rev5_event_study_narrow.csv", index=False)

# ---------- B ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc)
    for c in range(2014, 2026):
        if c == y: continue
        lc = value_at(c, q); dists = [(n, abs(n / lc - 1)) for n in notches(c)]; nn, dmin = min(dists, key=lambda t: t[1])
        rows.append(dict(notch=label, control_year=c, matched_revenue_lei=round(lc), nearest_notch_lei=round(nn), distance_pct=round(100 * dmin, 1), admissible=dmin > 0.25))
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/rev5_control_locations.csv", index=False)
print("\nB. closest calls among admissible controls (distance to nearest notch < 35%):"); print(B[B.admissible & (B.distance_pct < 35)].to_string(index=False))

# ---------- C ----------
def poly_b(values, thr, bw, half=0.3, excl=3, deg=5):
    edges = np.arange(thr * (1 - half), thr * (1 + half) + bw, bw); c, _ = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], bins=edges); c = c.astype(float)
    centers = edges[:-1] + bw / 2; x = (centers - thr) / bw; mask = (x < -excl) | (x > excl); cf = np.polyval(np.polyfit(x[mask], c[mask], deg), x); below = (x >= -excl) & (x < 0)
    return round((c[below].sum() - cf[below].sum()) / cf[below].mean(), 2), int(c.sum())
rows = []
for y in (2023, 2024, 2025):
    v = pp[(pp.year == y) & (pp.revenue > 0)].revenue.values; loc60 = 60e3 * fx[y - 1]
    b60, n60 = poly_b(v, loc60, 1500); b300, n300 = poly_b(v, 300_000, 1500)
    edges = np.arange(250_000, 450_001, 2_000); c, _ = np.histogram(v, bins=edges); k = np.argmax(c)
    rows.append(dict(year=y, eur60k_lei=round(loc60), b_at_60k=b60, b_at_300k=b300, firms_window=n60, modal_2k_bin_250_450k=f"{int(edges[k]):,}-{int(edges[k+1]):,}", modal_count=int(c[k]), median_count=int(np.median(c))))
C = pd.DataFrame(rows); print("\nC. EUR 60k rate boundary (1,500-lei bins, polynomial, window +/-30%):\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/rev5_60k_boundary.csv", index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, y in zip(axes, (2023, 2024, 2025)):
    v = pp[(pp.year == y) & (pp.revenue > 0)].revenue.values; edges = np.arange(250_000, 450_001, 1_000); c, _ = np.histogram(v, bins=edges)
    ax.bar((edges[:-1] + 500) / 1e3, c, width=0.9, color="#4C72B0"); loc60 = 60e3 * fx[y - 1]
    if y >= 2024: ax.axvline(loc60 / 1e3, color="orange", ls="--", lw=1.3, label=f"EUR 60,000 at prev. year-end rate = {loc60:,.0f} lei")
    ax.axvline(300, color="green", lw=1.2, label="VAT 300,000 lei" + (" (to Aug. 2025)" if y == 2025 else ""))
    if y == 2025: ax.axvline(395, color="green", ls="--", lw=1.2, label="VAT 395,000 lei (from Sept. 2025)")
    ax.set_title(str(y), loc="left"); ax.legend(fontsize=8); ax.set_ylabel("firms")
axes[-1].set_xlabel("total revenue, thousand lei (1,000-lei bins)"); fig.tight_layout(); fig.savefig(f"{OUT}/boundary_60k_2023_2025.png", dpi=130)

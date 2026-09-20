"""Feasibility check for the micro-enterprise threshold bunching paper.

1. Infer each firm-year's tax regime from the statements (tax paid / revenue vs tax / pre-tax profit).
2. Cross-check the inferred regime for 2023 against the ANAF tax-vector flags (IMP100 = profit tax, IMP120 = micro tax).
3. Draw revenue histograms around the legal threshold for every year 2014-2025, threshold converted at the
   BNR EUR/RON rate at the close of the previous fiscal year (the rate the in-year test uses), and compute a
   simple excess-mass statistic.
Outputs go to outputs/.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs"
os.makedirs(OUT, exist_ok=True)

# Legal threshold (EUR) applying to revenue realised in year t
THRESHOLD = {2014: 65_000, 2015: 65_000, 2016: 100_000, 2017: 500_000, 2018: 1_000_000, 2019: 1_000_000,
             2020: 1_000_000, 2021: 1_000_000, 2022: 1_000_000, 2023: 500_000, 2024: 500_000, 2025: 250_000}
SECOND_NOTCH = {2023: 60_000, 2024: 60_000, 2025: 60_000}  # 1% vs 3% rate boundary

fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]

p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet",
                    columns=["year", "form", "cui", "caen", "turnover", "revenue", "employees", "pretax", "net", "tax"])

# ---------- 1. regime inference ----------
rev = p["revenue"].where(p["revenue"] > 0)
p["rate_rev"] = p["tax"] / rev
p["rate_prof"] = p["tax"] / p["pretax"].where(p["pretax"] > 0)
micro_rates = [0.01, 0.02, 0.03]  # 1% / 2% (2017 only) / 3%
near_micro = np.zeros(len(p), bool)
for r in micro_rates:
    near_micro |= (p["rate_rev"] - r).abs() < 0.0035
near_profit = (p["rate_prof"] - 0.16).abs() < 0.025
has_tax = p["tax"] > 0
loss_but_taxed = has_tax & (p["pretax"] <= 0)          # only a revenue tax can produce this
p["regime"] = np.select(
    [~has_tax, loss_but_taxed, near_micro & ~near_profit, near_profit & ~near_micro, near_micro & near_profit],
    ["no_tax", "micro_loss", "micro", "profit", "ambiguous"], default="other")
print("Inferred regime shares by year (%):")
print((pd.crosstab(p["year"], p["regime"], normalize="index") * 100).round(1).to_string())

# ---------- 2. cross-check with ANAF flags, 2023 ----------
anaf_path = f"{ROOT}/data/raw/anaf/date_identificare_platitori_2023.csv"
if os.path.exists(anaf_path):
    a = pd.read_csv(anaf_path, sep="|", encoding="utf-8-sig", usecols=["COD_FISCAL", "TIP_CONTRIB", "IMP100", "IMP120"],
                    dtype=str, on_bad_lines="skip", engine="c", quoting=3, index_col=False)
    a["cui"] = pd.to_numeric(a["COD_FISCAL"], errors="coerce")
    a = a.dropna(subset=["cui"]).drop_duplicates("cui")
    a["cui"] = a["cui"].astype("int64")
    a["flag"] = np.select([(a.IMP120.str.strip() == "DA") & (a.IMP100.str.strip() != "DA"),
                           (a.IMP100.str.strip() == "DA") & (a.IMP120.str.strip() != "DA"),
                           (a.IMP100.str.strip() == "DA") & (a.IMP120.str.strip() == "DA")],
                          ["flag_micro", "flag_profit", "flag_both"], default="flag_none")
    m = p[p.year == 2023].merge(a[["cui", "flag"]], on="cui", how="left")
    m["flag"] = m["flag"].fillna("not_in_anaf")
    print(f"\n2023 statements: {len(m):,}; matched to ANAF June-2023 registry: {(m.flag != 'not_in_anaf').mean() * 100:.1f}%")
    ct = pd.crosstab(m["regime"], m["flag"])
    print("\nInferred regime (rows) vs ANAF flag (cols), 2023:")
    print(ct.to_string())
    core = m[m.regime.isin(["micro", "micro_loss", "profit"]) & m.flag.isin(["flag_micro", "flag_profit"])]
    agree = ((core.regime.isin(["micro", "micro_loss"]) & (core.flag == "flag_micro")) |
             ((core.regime == "profit") & (core.flag == "flag_profit"))).mean()
    print(f"\nAgreement where both measures are decisive: {agree * 100:.1f}% of {len(core):,} firms")
    ct.to_csv(f"{OUT}/regime_crosscheck_2023.csv")

# ---------- 3. histograms around the threshold ----------
def excess_mass(counts, centers, thr, bw, excl=3, deg=5):
    """Chetty et al. style: fit polynomial to bins outside [thr-excl*bw, thr+excl*bw], report excess below."""
    x = (centers - thr) / bw
    mask = (x < -excl) | (x > excl)
    coef = np.polyfit(x[mask], counts[mask], deg)
    cf = np.polyval(coef, x)
    below = (x >= -excl) & (x < 0)
    B = counts[below].sum() - cf[below].sum()
    return B / cf[below].mean() if cf[below].mean() > 0 else np.nan, cf

years = sorted(THRESHOLD)
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
stats = []
for ax, y in zip(axes.flat, years):
    thr = THRESHOLD[y]
    rate_prev, rate_cur = fx[y - 1], fx[y]
    d = p[(p.year == y) & (p.revenue > 0)]
    eur = d["revenue"] / rate_prev                      # revenue in EUR at previous year-end rate
    bw = {65_000: 1_000, 100_000: 2_000, 250_000: 5_000, 500_000: 10_000, 1_000_000: 20_000}[thr]
    lo, hi = thr * 0.5, thr * 1.5
    edges = np.arange(lo, hi + bw, bw)
    counts, _ = np.histogram(eur[(eur >= lo) & (eur < hi)], bins=edges)
    centers = edges[:-1] + bw / 2
    b, cf = excess_mass(counts.astype(float), centers, thr, bw)
    stats.append(dict(year=y, threshold_eur=thr, rate_prev=rate_prev, firms_in_window=int(counts.sum()), excess_mass_b=round(b, 2)))
    ax.bar(centers / 1000, counts, width=bw / 1000 * 0.9, color="#4C72B0")
    ax.plot(centers / 1000, cf, color="#C44E52", lw=1.2)
    ax.axvline(thr / 1000, color="k", ls="--", lw=1)
    ax.axvline(thr * rate_cur / rate_prev / 1000, color="grey", ls=":", lw=1)  # same threshold at current year-end rate
    if y in SECOND_NOTCH and lo < SECOND_NOTCH[y] < hi:
        ax.axvline(SECOND_NOTCH[y] / 1000, color="orange", ls="--", lw=1)
    ax.set_title(f"{y}: threshold €{thr:,}", fontsize=11)
    ax.set_xlabel("total revenue, € thousand (prev. year-end BNR rate)")
    ax.set_ylabel("firms")
fig.tight_layout()
fig.savefig(f"{OUT}/bunching_panels.png", dpi=130)
st = pd.DataFrame(stats)
print("\nExcess mass below threshold (b = excess firms in 3 bins below / counterfactual bin count):")
print(st.to_string(index=False))
st.to_csv(f"{OUT}/bunching_stats.csv", index=False)
p[["year", "cui", "regime", "rate_rev", "rate_prof"]].to_parquet(f"{ROOT}/data/panel/regime_inferred.parquet", index=False)
print("\nsaved", f"{OUT}/bunching_panels.png")

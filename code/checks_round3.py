"""Round three: anticipation of announced thresholds, the 2023 transition, and the Sept-2025 VAT change.

1. 2022: the cut to EUR 500k for 2023 was legislated in July 2022 (OG 16/2022). In-year threshold was still EUR 1M.
   A spike at 500k in 2022 revenue = firms targeting next year's status (Dec-31 rule), not the in-year rule.
2. 2025: the cut to EUR 100k for 2026 was legislated in Dec 2024 (OUG 156/2024). In-year threshold was 250k.
3. 2025 VAT threshold rose to 395,000 lei from 1 Sept 2025: look for a spike at 395k in 2025 turnover.
4. 2023 transition: where were the 2023 bunchers (revenue in the 3 bins under EUR 500k) in 2022?
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "turnover", "revenue"])

def excess(counts, centers, thr, bw, excl=3, deg=5):
    x = (centers - thr) / bw; mask = (x < -excl) | (x > excl)
    cf = np.polyval(np.polyfit(x[mask], counts[mask], deg), x); below = (x >= -excl) & (x < 0)
    return (counts[below].sum() - cf[below].sum()) / cf[below].mean(), cf

panels = [  # (year, column, threshold in lei, label, bin width)
    (2022, "revenue", 500_000 * fx[2022], "2022 revenue at €500k x end-2022 rate (announced Jul-2022 for 2023 status, deferred Dec-2022)", 10_000),
    (2021, "revenue", 500_000 * fx[2021], "placebo: 2021 revenue at €500k x end-2021 rate (no threshold there)", 10_000),
    (2025, "revenue", 100_000 * fx[2024], "2025 revenue at €100k x end-2024 rate (enacted Dec-2024 for 2026 status)", 2_000),
    (2024, "revenue", 100_000 * fx[2023], "placebo: 2024 revenue at €100k x end-2023 rate", 2_000),
    (2025, "turnover", 395_000, "2025 turnover at the new 395,000 lei VAT threshold (from Sept 2025)", 2_000),
    (2024, "turnover", 395_000, "placebo: 2024 turnover at 395,000 lei", 2_000),
]
fig, axes = plt.subplots(3, 2, figsize=(15, 12)); res = []
for ax, (y, col, thr, label, bw) in zip(axes.flat, panels):
    v = p[(p.year == y) & (p[col] > 0)][col].values
    edges = np.arange(thr * 0.6, thr * 1.4 + bw, bw); c, _ = np.histogram(v[(v >= edges[0]) & (v < edges[-1])], bins=edges)
    centers = edges[:-1] + bw / 2; b, cf = excess(c.astype(float), centers, thr, bw)
    ax.bar(centers / 1e3, c, width=bw / 1e3 * .9, color="#4C72B0"); ax.plot(centers / 1e3, cf, color="#C44E52", lw=1.2)
    ax.axvline(thr / 1e3, color="k", ls="--", lw=1); ax.set_title(f"{label}\nb = {b:.2f}", fontsize=9.5); ax.set_xlabel("thousand lei")
    res.append(dict(year=y, base=col, threshold_lei=round(thr), b=round(b, 2)))
fig.tight_layout()
fig.savefig(f"{OUT}/anticipation_checks.png", dpi=130)
print("1-3. Excess mass at announced / new thresholds vs placebos:"); print(pd.DataFrame(res).to_string(index=False))

# 4. 2023 transition
r23 = p[p.year == 2023].set_index("cui")["revenue"] / fx[2022]
r22 = p[p.year == 2022].set_index("cui")["revenue"] / fx[2021]
bunch = r23[(r23 >= 470_000) & (r23 < 500_000)].index
ctrl = r23[(r23 >= 400_000) & (r23 < 430_000)].index          # firms a bit further below, same year
def where_in_2022(ids):
    x = r22.reindex(ids); bins = [-1, 0, 400_000, 470_000, 500_000, 600_000, 1_000_000, 1e12]
    labels = ["no 2022 filing/zero", "<400k", "400-470k", "470-500k", "500-600k", "600k-1M", ">1M"]
    return pd.cut(x.fillna(-0.5), bins, labels=labels).value_counts(normalize=True).reindex(labels) * 100
t = pd.DataFrame({"2023 bunchers (470-500k)": where_in_2022(bunch), "2023 controls (400-430k)": where_in_2022(ctrl)}).round(1)
print(f"\n4. Where the 2023 firms were in 2022 (% of group; bunchers n={len(bunch):,}, controls n={len(ctrl):,}):"); print(t.to_string())
t.to_csv(f"{OUT}/transition_2023_origin.csv")
# same for 2024 bunchers, for comparison
r24 = p[p.year == 2024].set_index("cui")["revenue"] / fx[2023]
b24 = r24[(r24 >= 470_000) & (r24 < 500_000)].index; c24 = r24[(r24 >= 400_000) & (r24 < 430_000)].index
r22b = r23  # previous year for 2024 is 2023
def where_prev(ids, prev):
    x = prev.reindex(ids); bins = [-1, 0, 400_000, 470_000, 500_000, 600_000, 1_000_000, 1e12]
    labels = ["no prev filing/zero", "<400k", "400-470k", "470-500k", "500-600k", "600k-1M", ">1M"]
    return pd.cut(x.fillna(-0.5), bins, labels=labels).value_counts(normalize=True).reindex(labels) * 100
t2 = pd.DataFrame({"2024 bunchers (470-500k)": where_prev(b24, r23), "2024 controls (400-430k)": where_prev(c24, r23)}).round(1)
print(f"\n   Same for 2024 firms, position in 2023 (bunchers n={len(b24):,}, controls n={len(c24):,}):"); print(t2.to_string())

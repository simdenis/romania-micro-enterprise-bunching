"""(1) 2025 sub-notch: with VAT moved to 395k lei (Sept 2025), does a spike survive at the EUR 60k boundary (~298k-305k lei)?
   (2) Employee transitions in the panel around the 2023 one-employee rule."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "turnover", "revenue", "employees"])
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")

# (1) sub-notch
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True); bw = 1_000; res = []
for i, y in enumerate((2024, 2025)):
    for j, col in enumerate(("turnover", "revenue")):
        v = p[(p.year == y) & (p[col] > 0)][col].values
        edges = np.arange(250_000, 450_000 + bw, bw); c, _ = np.histogram(v[(v >= edges[0]) & (v < edges[-1])], bins=edges)
        centers = edges[:-1] + bw / 2; ax = axes[i, j]
        ax.bar(centers / 1e3, c, width=.9, color="#4C72B0")
        n60_prev, n60_cur = 60e3 * fx[y - 1], 60e3 * fx[y]
        ax.axvline(n60_prev / 1e3, color="orange", ls="--", lw=1.2, label=f"€60k at prev. year-end rate = {n60_prev:,.0f}")
        ax.axvline(n60_cur / 1e3, color="orange", ls=":", lw=1.2, label=f"€60k at current year-end rate = {n60_cur:,.0f}")
        ax.axvline(300, color="green", lw=1.2, label="300,000 lei (VAT to Aug-2025)")
        if y == 2025: ax.axvline(395, color="green", ls="--", lw=1.2, label="395,000 lei (VAT from Sept-2025)")
        ax.set_title(f"{y}: {col}", loc="left", fontsize=10); ax.legend(fontsize=7.5); ax.set_ylabel("firms")
        def band(a, b): return int(c[(centers > a) & (centers < b)].sum())
        res.append(dict(year=y, base=col, f295_300k=band(295e3, 300e3), f300_305k=band(300e3, 305e3), f305_310k=band(305e3, 310e3),
                        f390_395k=band(390e3, 395e3), f395_400k=band(395e3, 400e3), f380_385k=band(380e3, 385e3)))
for ax in axes[-1]: ax.set_xlabel("thousand lei (1,000 lei bins)")
fig.suptitle("2025: VAT threshold moved to 395,000 lei; does the €60k boundary keep its own spike?", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(f"{OUT}/subnotch_2025.png", dpi=130)
print("(1) firms per 5k-lei band:"); print(pd.DataFrame(res).to_string(index=False))

# (2) employee transitions: firms with 0 employees in year t, where are they in t+1?
d = p.merge(reg[["year", "cui", "regime"]], on=["year", "cui"]).drop_duplicates(["year", "cui"])
d["micro"] = d.regime.isin(["micro", "micro_loss"]); d["emp"] = d.employees.fillna(0)
w = d.pivot(index="cui", columns="year", values=["emp", "micro", "revenue"])
out = []
for t in (2019, 2020, 2021, 2022, 2023, 2024):
    base = w[(w[("micro", t)] == True) & (w[("emp", t)] == 0) & (w[("revenue", t)] > 0)]
    nxt_emp, nxt_micro, nxt_rev = base[("emp", t + 1)], base[("micro", t + 1)], base[("revenue", t + 1)]
    filed = nxt_rev.notna()
    row = dict(base_year=t, firms_micro_0emp=len(base),
               no_filing_next=round(100 * (~filed).mean(), 1),
               stay_micro_1emp=round(100 * (filed & (nxt_micro == True) & (nxt_emp == 1)).mean(), 1),
               stay_micro_2plus=round(100 * (filed & (nxt_micro == True) & (nxt_emp >= 2)).mean(), 1),
               stay_micro_0emp=round(100 * (filed & (nxt_micro == True) & (nxt_emp == 0)).mean(), 1),
               leave_micro=round(100 * (filed & (nxt_micro != True)).mean(), 1))
    out.append(row)
t2 = pd.DataFrame(out); print("\n(2) Micro firms with zero employees in year t: status in t+1 (% of group):"); print(t2.to_string(index=False))
t2.to_csv(f"{OUT}/employee_transitions.csv", index=False)
# implied hires: excess 0->1 moves in 2022->2023 relative to the 2019-2021 average rate
base_rate = t2[t2.base_year.isin([2019, 2020, 2021])].stay_micro_1emp.mean()
n22 = t2.loc[t2.base_year == 2022, "firms_micro_0emp"].item(); r22 = t2.loc[t2.base_year == 2022, "stay_micro_1emp"].item()
print(f"\n0->1 employee among zero-employee micro firms: {r22:.1f}% in 2022->2023 vs {base_rate:.1f}% average 2019-21 -> "
      f"excess {(r22 - base_rate):.1f} pp x {n22:,} firms = {(r22 - base_rate) / 100 * n22:,.0f} extra one-employee firms")
# same for all firms with zero employees (regime-free)
out2 = []
for t in (2019, 2020, 2021, 2022, 2023, 2024):
    base = w[(w[("emp", t)] == 0) & (w[("revenue", t)] > 0) & (w[("revenue", t)] < 500e3 * fx[t - 1])]
    nxt_emp, nxt_rev = base[("emp", t + 1)], base[("revenue", t + 1)]; filed = nxt_rev.notna()
    out2.append(dict(base_year=t, firms_0emp=len(base), no_filing_next=round(100 * (~filed).mean(), 1),
                     to_1emp=round(100 * (filed & (nxt_emp == 1)).mean(), 1), to_2plus=round(100 * (filed & (nxt_emp >= 2)).mean(), 1),
                     stay_0emp=round(100 * (filed & (nxt_emp == 0)).mean(), 1)))
t3 = pd.DataFrame(out2); print("\n(2b) All firms with zero employees and revenue < €500k in t: employees in t+1 (%):"); print(t3.to_string(index=False))
t3.to_csv(f"{OUT}/employee_transitions_all.csv", index=False)

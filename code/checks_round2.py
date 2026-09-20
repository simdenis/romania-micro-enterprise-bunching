"""Round-two checks on the micro-enterprise threshold bunching.

A. Robustness of the excess-mass statistic b: bin width x polynomial degree x excluded window, plus bootstrap SE.
B. Lei-tracking zoom: revenue in lei around the euro threshold for adjacent years; the spike should sit at
   threshold x BNR rate and move with the rate. Also tells which rate firms target (previous vs current year-end).
C. The 60k EUR sub-notch (1% vs 3%, from 2023) versus the 300,000 lei VAT threshold, in lei with 1k bins.
D. VAT registration threshold bunching on turnover: 220,000 lei (to 2017) and 300,000 lei (from 2018).
E. Employee margin: distribution of employees among micro-window firms, before and after the 2023 one-employee rule.
Outputs in outputs/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs"
THRESHOLD = {2014: 65_000, 2015: 65_000, 2016: 100_000, 2017: 500_000, 2018: 1_000_000, 2019: 1_000_000,
             2020: 1_000_000, 2021: 1_000_000, 2022: 1_000_000, 2023: 500_000, 2024: 500_000, 2025: 250_000}
VAT_LEI = {y: (220_000 if y <= 2017 else 300_000) for y in range(2014, 2026)}
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet",
                    columns=["year", "cui", "turnover", "revenue", "employees", "tax", "pretax"])
rng = np.random.default_rng(7)


def excess(counts, centers, thr, bw, excl, deg):
    x = (centers - thr) / bw
    mask = (x < -excl) | (x > excl)
    cf = np.polyval(np.polyfit(x[mask], counts[mask], deg), x)
    below = (x >= -excl) & (x < 0)
    return (counts[below].sum() - cf[below].sum()) / cf[below].mean()


def hist(values, thr, bw, half=0.5):
    edges = np.arange(thr * (1 - half), thr * (1 + half) + bw, bw)
    c, _ = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], bins=edges)
    return c.astype(float), edges[:-1] + bw / 2


# ---------- A. robustness ----------
rows = []
for y, thr in THRESHOLD.items():
    d = p[(p.year == y) & (p.revenue > 0)]
    eur = (d["revenue"] / fx[y - 1]).values
    base_bw = {65_000: 1_000, 100_000: 2_000, 250_000: 5_000, 500_000: 10_000, 1_000_000: 20_000}[thr]
    for bwm in (0.5, 1, 2):
        bw = base_bw * bwm
        counts, centers = hist(eur, thr, bw)
        for deg in (3, 5, 7):
            for excl in (2, 3, 5):
                rows.append(dict(year=y, bw=bw, deg=deg, excl=excl, b=excess(counts, centers, thr, bw, excl, deg)))
    # bootstrap at baseline spec: resample bin counts (multinomial) 300 times
    counts, centers = hist(eur, thr, base_bw)
    bs = [excess(rng.multinomial(int(counts.sum()), counts / counts.sum()).astype(float), centers, thr, base_bw, 3, 5)
          for _ in range(300)]
    rows.append(dict(year=y, bw=base_bw, deg=5, excl=3, b=excess(counts, centers, thr, base_bw, 3, 5), boot_se=np.std(bs)))
rob = pd.DataFrame(rows)
summary = rob.groupby("year").agg(b_baseline=("b", lambda s: s.iloc[-1]), b_min=("b", "min"), b_max=("b", "max"),
                                  b_median=("b", "median"), boot_se=("boot_se", "max"))
summary["threshold"] = pd.Series(THRESHOLD)
print("A. Robustness of b across 27 specs (bin width x degree x excluded window), plus bootstrap SE at baseline:")
print(summary.round(2).to_string())
rob.to_csv(f"{OUT}/robustness_b_all_specs.csv", index=False)
summary.round(3).to_csv(f"{OUT}/robustness_b_summary.csv")

# ---------- B. lei-tracking zoom ----------
fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
bw = 10_000
loc = []
for ax, y in zip(axes, range(2018, 2023)):
    d = p[(p.year == y) & (p.revenue > 0)]["revenue"].values
    edges = np.arange(4_300_000, 5_300_000 + bw, bw)
    c, _ = np.histogram(d[(d >= edges[0]) & (d < edges[-1])], bins=edges)
    centers = edges[:-1] + bw / 2
    t_prev, t_cur = 1_000_000 * fx[y - 1], 1_000_000 * fx[y]
    ax.bar(centers / 1e6, c, width=bw / 1e6 * 0.9, color="#4C72B0")
    ax.axvline(t_prev / 1e6, color="k", ls="--", lw=1.2, label=f"€1M at prev. year-end rate = {t_prev:,.0f} lei")
    ax.axvline(t_cur / 1e6, color="#C44E52", ls=":", lw=1.5, label=f"€1M at current year-end rate = {t_cur:,.0f} lei")
    ax.set_title(f"{y}", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylabel("firms")
    # modal bin among bins within 150k lei of either candidate
    win = (centers > min(t_prev, t_cur) - 150_000) & (centers < max(t_prev, t_cur) + 150_000)
    mode_center = centers[win][np.argmax(c[win])]
    loc.append(dict(year=y, thr_prev_rate=round(t_prev), thr_cur_rate=round(t_cur), modal_bin_center=round(mode_center),
                    dist_to_prev=round(mode_center - t_prev), dist_to_cur=round(mode_center - t_cur)))
axes[-1].set_xlabel("total revenue, million lei (10,000 lei bins)")
fig.tight_layout()
fig.savefig(f"{OUT}/lei_tracking_1M_2018_2022.png", dpi=130)
loc = pd.DataFrame(loc)
print("\nB. Modal bin location (lei) vs the two candidate thresholds (negative = spike below):")
print(loc.to_string(index=False))
loc.to_csv(f"{OUT}/lei_tracking_1M.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
bw = 5_000
for ax, y in zip(axes, (2023, 2024)):
    d = p[(p.year == y) & (p.revenue > 0)]["revenue"].values
    edges = np.arange(2_200_000, 2_800_000 + bw, bw)
    c, _ = np.histogram(d[(d >= edges[0]) & (d < edges[-1])], bins=edges)
    centers = edges[:-1] + bw / 2
    t_prev, t_cur = 500_000 * fx[y - 1], 500_000 * fx[y]
    ax.bar(centers / 1e6, c, width=bw / 1e6 * 0.9, color="#4C72B0")
    ax.axvline(t_prev / 1e6, color="k", ls="--", lw=1.2, label=f"€500k at prev. year-end rate = {t_prev:,.0f} lei")
    ax.axvline(t_cur / 1e6, color="#C44E52", ls=":", lw=1.5, label=f"€500k at current year-end rate = {t_cur:,.0f} lei")
    ax.axvline(2.5, color="grey", ls="-", lw=0.8, label="2,500,000 lei (round number)")
    ax.set_title(f"{y}", loc="left"); ax.legend(fontsize=9); ax.set_ylabel("firms")
axes[-1].set_xlabel("total revenue, million lei (5,000 lei bins)")
fig.tight_layout()
fig.savefig(f"{OUT}/lei_tracking_500k_2023_2024.png", dpi=130)

# ---------- C. 60k EUR sub-notch vs 300k lei VAT threshold ----------
fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex=True)
bw = 1_000
c60 = []
for i, y in enumerate(range(2021, 2026)):
    d = p[(p.year == y)]
    for j, (col, label) in enumerate((("turnover", "net turnover (VAT base)"), ("revenue", "total revenue (micro-tax base)"))):
        v = d[d[col] > 0][col].values
        edges = np.arange(240_000, 360_000 + bw, bw)
        c, _ = np.histogram(v[(v >= edges[0]) & (v < edges[-1])], bins=edges)
        centers = edges[:-1] + bw / 2
        ax = axes[i, j]
        ax.bar(centers / 1e3, c, width=bw / 1e3 * 0.9, color="#4C72B0")
        ax.axvline(300, color="green", ls="-", lw=1.2, label="VAT threshold 300,000 lei")
        if y >= 2024:
            n60 = 60_000 * fx[y - 1]
            ax.axvline(n60 / 1e3, color="orange", ls="--", lw=1.2, label=f"€60k notch = {n60:,.0f} lei")
        ax.set_title(f"{y}: {label}", loc="left", fontsize=10); ax.legend(fontsize=8); ax.set_ylabel("firms")
        # counts in the 3 bins just below 300k vs 3 bins just above
        below = c[(centers > 297_000) & (centers < 300_000)].sum(); above = c[(centers > 300_000) & (centers < 303_000)].sum()
        c60.append(dict(year=y, base=col, firms_297_300k=int(below), firms_300_303k=int(above), ratio=round(below / max(above, 1), 2)))
for ax in axes[-1]: ax.set_xlabel("thousand lei (1,000 lei bins)")
fig.suptitle("300,000 lei VAT threshold and the €60,000 micro sub-notch (from 2023), 2021-2025", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/notch_60k_vs_vat_300k.png", dpi=130)
c60 = pd.DataFrame(c60)
print("\nC. Firms just below vs just above 300,000 lei (3k-lei bands):")
print(c60.to_string(index=False))

# ---------- D. VAT threshold bunching on turnover, all years ----------
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
vat = []
for ax, y in zip(axes.flat, range(2014, 2026)):
    thr = VAT_LEI[y]; bw = 2_000
    v = p[(p.year == y) & (p.turnover > 0)]["turnover"].values
    counts, centers = hist(v, thr, bw, half=0.4)
    b = excess(counts, centers, thr, bw, 3, 5)
    x = (centers - thr) / bw; mask = (x < -3) | (x > 3)
    cf = np.polyval(np.polyfit(x[mask], counts[mask], 5), x)
    ax.bar(centers / 1e3, counts, width=bw / 1e3 * 0.9, color="#55A868"); ax.plot(centers / 1e3, cf, color="#C44E52", lw=1.2)
    ax.axvline(thr / 1e3, color="k", ls="--", lw=1)
    micro_lei = THRESHOLD[y] * fx[y - 1]
    if thr * 0.6 < micro_lei < thr * 1.4:
        ax.axvline(micro_lei / 1e3, color="orange", ls="--", lw=1, label="micro threshold")
        ax.legend(fontsize=8)
    ax.set_title(f"{y}: VAT threshold {thr:,} lei (b = {b:.2f})", fontsize=11)
    ax.set_xlabel("net turnover, thousand lei"); ax.set_ylabel("firms")
    vat.append(dict(year=y, vat_threshold_lei=thr, firms_in_window=int(counts.sum()), excess_mass_b=round(b, 2)))
fig.tight_layout()
fig.savefig(f"{OUT}/vat_threshold_panels.png", dpi=130)
vat = pd.DataFrame(vat)
print("\nD. VAT threshold excess mass:")
print(vat.to_string(index=False))
vat.to_csv(f"{OUT}/vat_bunching_stats.csv", index=False)

# ---------- E. employee margin ----------
d = p[(p.revenue > 0) & (p.year >= 2019)].copy()
d["rev_eur"] = d["revenue"] / d["year"].map(lambda y: fx[y - 1])
d = d[d.rev_eur < 500_000]
d["emp_bin"] = pd.cut(d["employees"].fillna(0), [-1, 0, 1, 2, 4, 1e9], labels=["0", "1", "2", "3-4", "5+"])
emp = pd.crosstab(d["year"], d["emp_bin"], normalize="index") * 100
emp["firms"] = d.groupby("year").size()
print("\nE. Employees among firms with revenue < €500k (% of firms):")
print(emp.round(1).to_string())
emp.round(2).to_csv(f"{OUT}/employee_margin.csv")
# same, restricted to firms that were micro (tax-inferred) in the year
reg = pd.read_parquet(f"{ROOT}/data/panel/regime_inferred.parquet")
d2 = d.merge(reg[["year", "cui", "regime"]], on=["year", "cui"])
d2 = d2[d2.regime.isin(["micro", "micro_loss"])]
emp2 = pd.crosstab(d2["year"], d2["emp_bin"], normalize="index") * 100
emp2["firms"] = d2.groupby("year").size()
print("\nE2. Same, firms inferred to be on the micro regime that year (% of firms):")
print(emp2.round(1).to_string())
emp2.round(2).to_csv(f"{OUT}/employee_margin_micro_only.csv")
fig, ax = plt.subplots(figsize=(9, 5))
for k in ["0", "1", "2"]:
    ax.plot(emp.index, emp[k], marker="o", label=f"{k} employees")
ax.axvline(2022.5, color="grey", ls="--", lw=1); ax.text(2022.55, ax.get_ylim()[1] * 0.95, "one-employee rule (2023)", fontsize=9)
ax.set_ylabel("% of firms with revenue < €500k"); ax.set_xlabel("year"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/employee_margin.png", dpi=130)
print("\nfigures saved in", OUT)

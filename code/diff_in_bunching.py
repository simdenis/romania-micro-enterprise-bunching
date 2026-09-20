"""Difference-in-bunching estimator.

For each notch (year t, threshold T_t in lei = EUR threshold x BNR rate at the close of t-1), the counterfactual
density near T_t is the density at the SAME QUANTILE of the revenue distribution in control years where no
threshold (micro, year-end eligibility, VAT, 60k sub-notch) lies within the window. No polynomial is fitted.

Bins are relative to the notch location (1% of the threshold wide), window +/-20%. Shares are counts in a bin
divided by firms in the window, so scale differences across years drop out. Excess mass
    b = sum_{k in B} (s_t,k - sbar_c,k) / mean_{k in B} sbar_c,k ,   B = the 5 bins just below the notch,
i.e. excess firms in the bunching region expressed in units of counterfactual firms per bin (Chetty et al. b).
Missing mass M is the same sum over the 5 bins just above. Bootstrap SE from multinomial resampling of
treated and control bin counts. Spec range: bins of 0.5/1/2 %, bunching region 3/5/8 % of threshold.
"""
import os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "turnover", "revenue"])
rev = {y: np.sort(g["revenue"].values[g["revenue"].values > 0]) for y, g in p.groupby("year")}
rng = np.random.default_rng(11)

MICRO = {2014: 65e3, 2015: 65e3, 2016: 100e3, 2017: 500e3, 2018: 1e6, 2019: 1e6, 2020: 1e6, 2021: 1e6, 2022: 1e6,
         2023: 500e3, 2024: 500e3, 2025: 250e3}
# every notch on REVENUE known to exist in a year, in lei: in-year micro threshold (prev year-end rate),
# year-end eligibility threshold for next year when it differed and was already law, 60k sub-notch (2023+)
def notches(y):
    n = [MICRO[y] * fx[y - 1]]
    if y == 2022: n.append(500e3 * fx[2022])       # OG 16/2022 announced 500k for 2023 status (deferred by Legea 370/2022 on 20 Dec 2022)
    if y == 2025: n.append(100e3 * fx[2024]); n.append(100e3 * fx[2025])   # OUG 156/2024: 2026 status set by 2025 revenue; firms can only know the end-2024 rate
    if y >= 2024: n.append(60e3 * fx[y - 1])       # 1% vs 3% rate boundary (Legea 296/2023; 2023 was a flat 1%)
    n.append(220e3 if y <= 2017 else (300e3 if y <= 2024 else 395e3))   # VAT threshold (on turnover, but close)
    if y == 2025: n.append(300e3)
    return n

# notches to estimate: (label, year, location in lei)
TARGETS = [(f"{y}: micro €{int(MICRO[y]):,}", y, MICRO[y] * fx[y - 1]) for y in MICRO]
TARGETS += [("2022: announced-for-2023 €500,000", 2022, 500e3 * fx[2022]),
            ("2025: next-year eligibility €100,000 (end-2024 rate)", 2025, 100e3 * fx[2024])]

def quantile_of(y, x):            # F_y(x)
    return np.searchsorted(rev[y], x) / len(rev[y])
def value_at(y, q):               # F_y^{-1}(q)
    return rev[y][min(int(q * len(rev[y])), len(rev[y]) - 1)]

def rel_hist(y, loc, bw, half=0.20):
    edges = loc * (1 + np.arange(-half, half + bw / 2, bw))
    c, _ = np.histogram(rev[y], bins=edges)
    return c.astype(float)

def estimate(y, loc, controls, bw=0.01, region=0.05, half=0.20):
    nb = int(round(region / bw)); k0 = int(round(half / bw))          # index of first bin at/after the notch
    ct = rel_hist(y, loc, bw, half); st = ct / ct.sum()
    ccs = []
    for c in controls:
        loc_c = value_at(c, quantile_of(y, loc))                      # same quantile in control year
        cc = rel_hist(c, loc_c, bw, half); ccs.append(cc)
    sc = np.mean([cc / cc.sum() for cc in ccs], axis=0)
    B = slice(k0 - nb, k0); A = slice(k0, k0 + nb)
    b = (st[B] - sc[B]).sum() / sc[B].mean()
    m = (st[A] - sc[A]).sum() / sc[A].mean()
    return b, m, st, sc, ct, ccs

def bootstrap(ct, ccs, bw, region, half, reps=300):
    nb = int(round(region / bw)); k0 = int(round(half / bw)); out = []
    for _ in range(reps):
        st = rng.multinomial(int(ct.sum()), ct / ct.sum()) / ct.sum()
        sc = np.mean([rng.multinomial(int(cc.sum()), cc / cc.sum()) / cc.sum() for cc in ccs], axis=0)
        B = slice(k0 - nb, k0); out.append((st[B] - sc[B]).sum() / sc[B].mean())
    return np.std(out)

rows = []; fig, axes = plt.subplots(5, 3, figsize=(16, 20)); axes = axes.flat
for label, y, loc in TARGETS:
    q = quantile_of(y, loc)
    controls = []
    for c in range(2014, 2026):
        if c == y: continue
        loc_c = value_at(c, q)
        if all(abs(n / loc_c - 1) > 0.25 for n in notches(c)):        # no known notch within +/-25% of the matched location
            controls.append(c)
    if not controls: print("no controls for", label); continue
    b, m, st, sc, ct, ccs = estimate(y, loc, controls)
    se = bootstrap(ct, ccs, 0.01, 0.05, 0.20)
    specs = [estimate(y, loc, controls, bw, region)[0] for bw, region in itertools.product((0.005, 0.01, 0.02), (0.03, 0.05, 0.08))]
    rows.append(dict(notch=label, year=y, location_lei=round(loc), quantile=round(q, 4), controls=",".join(map(str, controls)),
                     firms_in_window=int(ct.sum()), b_diff=round(b, 2), boot_se=round(se, 2), missing_mass_above=round(m, 2),
                     b_spec_min=round(min(specs), 2), b_spec_max=round(max(specs), 2)))
    ax = next(axes); x = (np.arange(len(st)) - 20) + 0.5
    ax.bar(x, st * 100, width=0.9, color="#4C72B0", label=f"{y} (treated)")
    ax.step(x - 0.5, sc * 100, where="post", color="#C44E52", lw=1.4, label=f"controls {controls[0]}-{controls[-1]} (same quantile)")
    ax.axvline(0, color="k", ls="--", lw=1); ax.set_title(f"{label}\nb = {b:.2f} (bootstrap SE {se:.2f})", fontsize=9.5)
    ax.set_xlabel("% distance from notch (1% bins)"); ax.set_ylabel("% of firms in ±20% window"); ax.legend(fontsize=7.5)
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/diff_in_bunching.png", dpi=120)
res = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(res.drop(columns=["quantile"]).to_string(index=False)); res.to_csv(f"{OUT}/diff_in_bunching.csv", index=False)

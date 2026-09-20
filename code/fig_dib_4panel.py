"""Four-panel version of the difference-in-bunching figure for the body of the paper."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
want = ["2016: micro €100,000", "2021: micro €1,000,000", "2023: micro €500,000", "2025: next-year eligibility €100,000 (end-2024 rate)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8)); axes = axes.flat
for label, y, loc in TARGETS:
    if label not in want: continue
    q = quantile_of(y, loc); ctr = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
    b, m, st, sc, ct, ccs = estimate(y, loc, ctr); ax = next(axes); x = (np.arange(len(st)) - 20) + 0.5
    ax.bar(x, st * 100, width=0.9, color="#4C72B0", label=f"{y} (treated)"); ax.step(x - 0.5, sc * 100, where="post", color="#C44E52", lw=1.4, label="same-quantile controls")
    ax.axvline(0, color="k", ls="--", lw=1); ax.set_title(label.replace(" (end-2024 rate)", ""), fontsize=10); ax.set_xlabel("% distance from notch (1% bins)"); ax.set_ylabel("% of firms in ±20% window"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/diff_in_bunching_4panel.png", dpi=130); print("saved")

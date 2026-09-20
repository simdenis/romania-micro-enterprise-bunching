"""Scale-free summary of the difference-in-bunching results: excess FIRMS in the region just below each notch
(invariant to bin width), as a count and as % of firms in the +/-20% window, for regions of 3, 5 and 8% of the threshold."""
import importlib.util, os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("dib", f"{ROOT}/scripts/diff_in_bunching.py")
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]      # reuse data + functions only
exec(compile(src, "dib", "exec"))
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc)
    controls = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
    r = dict(notch=label)
    for region in (0.03, 0.05, 0.08):
        b, m, st, sc, ct, ccs = estimate(y, loc, controls, bw=0.01, region=region)
        nb = int(round(region / 0.01)); k0 = 20; N = ct.sum()
        excess = (st[k0 - nb:k0] - sc[k0 - nb:k0]).sum() * N
        r[f"excess_firms_{int(region*100)}pct"] = int(round(excess)); r[f"excess_share_{int(region*100)}pct"] = round(100 * excess / N, 2)
    r["firms_in_window"] = int(ct.sum()); rows.append(r)
res = pd.DataFrame(rows); pd.set_option("display.width", 250); print(res.to_string(index=False)); res.to_csv(f"{OUT}/diff_in_bunching_counts.csv", index=False)

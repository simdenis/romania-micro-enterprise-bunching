"""Re-estimate every notch with a stricter admissibility rule (no known notch within 35% instead of 25% of the matched location)."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc)
    c25 = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]
    c35 = [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.35 for n in notches(c))]
    b25 = estimate(y, loc, c25)[0]; b35 = estimate(y, loc, c35)[0] if c35 else np.nan
    rows.append(dict(notch=label, controls_25=len(c25), controls_35=len(c35), b_25=round(b25, 2), b_35=round(b35, 2) if b35 == b35 else np.nan, dropped=",".join(str(c) for c in c25 if c not in c35)))
R = pd.DataFrame(rows); pd.set_option("display.width", 200); print(R.to_string(index=False)); R.to_csv(f"{OUT}/strict_controls.csv", index=False)

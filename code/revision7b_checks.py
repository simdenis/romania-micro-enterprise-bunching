"""Count-based placebo distribution (net window difference and 5-bin excess at non-notch locations), conditional employee
transitions, and the 2015 year-end notch (EUR 100k for 2016 status, OUG 50/2015 of Oct 2015) as a notch in its own right."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
N = {y: len(rev[y]) for y in rev}
def notches2(y):                    # add the 2015 year-end notch to the exclusion list
    n = notches(y)
    if y == 2015: n.append(100e3 * fx[2015])
    return n
def controls_for(y, q): return [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches2(c))]
def count_based(y, loc):
    q = quantile_of(y, loc); ctr = controls_for(y, q)
    if len(ctr) < 3: return None
    ct = rel_hist(y, loc, 0.01, 0.20); st = ct / N[y]; sc = np.mean([rel_hist(c, value_at(c, q), 0.01, 0.20) / N[c] for c in ctr], axis=0)
    diff = (st - sc) * N[y]; return dict(b_count=(st[15:20] - sc[15:20]).sum() / sc[15:20].mean(), excess5=diff[15:20].sum(), deficit20=diff[20:40].sum(), net=diff.sum(), net_pct_of_window=100 * diff.sum() / ct.sum())
rows = []
for y in range(2014, 2026):
    for mult in (0.45, 0.6, 0.75, 1.35, 1.6, 2.0, 2.5):
        loc = MICRO[y] * fx[y - 1] * mult
        if any(abs(n / loc - 1) <= 0.25 for n in notches2(y)): continue
        r = count_based(y, loc)
        if r: rows.append(dict(year=y, mult=mult, location=round(loc), **{k: round(v, 2) for k, v in r.items()}))
P = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"count-based placebos (n={len(P)}): b_count mean {P.b_count.mean():.2f} sd {P.b_count.std():.2f} p95|b| {P.b_count.abs().quantile(.95):.2f} max {P.b_count.abs().max():.2f}; net window %: mean {P.net_pct_of_window.mean():.1f} sd {P.net_pct_of_window.std():.1f} p95|.| {P.net_pct_of_window.abs().quantile(.95):.1f}")
P.to_csv(f"{OUT}/rev7_count_based_placebo.csv", index=False)
# window-normalised placebo recomputed with the 2015 year-end notch excluded
rows = []
for y in range(2014, 2026):
    for mult in (0.45, 0.6, 0.75, 1.35, 1.6, 2.0, 2.5):
        loc = MICRO[y] * fx[y - 1] * mult
        if any(abs(n / loc - 1) <= 0.25 for n in notches2(y)): continue
        q = quantile_of(y, loc); ctr = controls_for(y, q)
        if len(ctr) < 3: continue
        rows.append(dict(year=y, mult=mult, b=round(estimate(y, loc, ctr)[0], 3)))
P2 = pd.DataFrame(rows); ab = P2.b.abs()
print(f"window placebos without the 2015 year-end notch (n={len(P2)}): mean {P2.b.mean():.3f} sd {P2.b.std():.3f} p90 {ab.quantile(.9):.2f} p95 {ab.quantile(.95):.2f} p99 {ab.quantile(.99):.2f} max {ab.max():.2f}")
P2.to_csv(f"{OUT}/rev7_placebo_clean.csv", index=False)
# the 2015 year-end notch as an estimate
loc15 = 100e3 * fx[2015]; q = quantile_of(2015, loc15); ctr = controls_for(2015, q); b15, m15, st, sc, ct, ccs = estimate(2015, loc15, ctr)
print(f"2015 year-end EUR 100k notch (for 2016 status, at end-2015 rate {loc15:,.0f} lei): b = {b15:.2f}, m = {m15:.2f}, controls {ctr}, firms in window {int(ct.sum()):,}, excess firms {(st[15:20]-sc[15:20]).sum()*ct.sum():,.0f}")
# conditional employee transitions (direct)
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "employees"]).drop_duplicates(["year", "cui"])
pp["emp"] = pp.employees.fillna(0); pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y - 1])
w = pp.pivot(index="cui", columns="year", values=["emp", "rev_eur"]); rows = []
for t in range(2019, 2025):
    base = w[(w[("emp", t)] == 0) & (w[("rev_eur", t)] > 0) & (w[("rev_eur", t)] < 500e3)]; nxt = base[("emp", t + 1)]; filed = base[("rev_eur", t + 1)].notna(); n = filed.sum()
    rows.append(dict(base_year=t, firms_0emp_filing_next=int(n), to_1emp=round(100 * (nxt[filed] == 1).mean(), 1), to_2plus=round(100 * (nxt[filed] >= 2).mean(), 1), stay_0emp=round(100 * (nxt[filed] == 0).mean(), 1)))
E = pd.DataFrame(rows); print(E.to_string(index=False)); E.to_csv(f"{OUT}/employee_transitions_conditional.csv", index=False)

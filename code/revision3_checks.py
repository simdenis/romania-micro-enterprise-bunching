"""Third round: (A) year-block bootstrap for the DiB estimates; (B) reduced-form elasticities with stated assumptions,
reported even where the assumptions fail; (C) sole-shareholder naming check in the trade register."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
src = open(f"{ROOT}/scripts/diff_in_bunching.py").read().split("rows = []")[0]; exec(compile(src, "dib", "exec"))
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax"])
rng = np.random.default_rng(21)
def controls_for(y, q): return [c for c in range(2014, 2026) if c != y and all(abs(n / value_at(c, q) - 1) > 0.25 for n in notches(c))]

# ---------- A. year-block bootstrap ----------
rows = []
for label, y, loc in TARGETS:
    q = quantile_of(y, loc); ctr = controls_for(y, q); b, m, st, sc, ct, ccs = estimate(y, loc, ctr)
    shares = [cc / cc.sum() for cc in ccs]; bs = []
    for _ in range(500):
        idx = rng.integers(0, len(ctr), len(ctr))                     # resample control YEARS with replacement
        sc_b = np.mean([shares[i] for i in idx], axis=0)
        st_b = rng.multinomial(int(ct.sum()), ct / ct.sum()) / ct.sum()  # and treated firms
        bs.append((st_b[15:20] - sc_b[15:20]).sum() / sc_b[15:20].mean())
    rows.append(dict(notch=label, b=round(b, 2), n_controls=len(ctr), block_boot_se=round(np.std(bs), 2), block_boot_p2_5=round(np.percentile(bs, 2.5), 2), block_boot_p97_5=round(np.percentile(bs, 97.5), 2)))
A = pd.DataFrame(rows); pd.set_option("display.width", 250); print("A. Year-block bootstrap (control years resampled with replacement, 500 draws):\n" + A.to_string(index=False)); A.to_csv(f"{OUT}/rev3_block_bootstrap.csv", index=False)

# ---------- B. elasticities ----------
# Kleven-Waseem reduced form for a notch that raises the average tax rate by dt at z*: with 1% bins, b ~ dz*/z* in percent
# (Saez approximation, excess mass = counterfactual density x dz*). Then e = (dz*/z*)^2 / (2 dt/(1 - t)).
# dt for this notch = 0.16 x margin - tau (profit tax on the margin minus the revenue tax), evaluated at (i) the median
# reported margin of bunchers and (ii) a fixed 20% margin. Where dt <= 0 the notch is not a tax increase and e is undefined.
RATE = {2014: .03, 2015: .03, 2016: .03, 2017: .01, 2018: .01, 2019: .01, 2020: .01, 2021: .01, 2022: .01, 2023: .01, 2024: .03, 2025: .03}
qt = pd.read_csv(f"{OUT}/rev2_quantities.csv"); rows = []
for _, r in qt.iterrows():
    y = int(r.year); tau = RATE[y]; dz_b = r.b / 100.0          # b with 1% bins -> dz*/z* (share)
    dz_hole = r.hole_width_pct / 100.0 if r.hole_width_pct == r.hole_width_pct else np.nan
    for margin_lab, margin in [("median buncher margin", r.median_margin_bunchers), ("margin 20%", 0.20)]:
        dt = 0.16 * margin - tau
        e_b = (dz_b ** 2) / (2 * dt / (1 - tau)) if dt > 0 else np.nan
        e_h = (dz_hole ** 2) / (2 * dt / (1 - tau)) if (dt > 0 and dz_hole == dz_hole) else np.nan
        rows.append(dict(notch=r.notch, year=y, tau=tau, margin_basis=margin_lab, margin=round(margin, 3), dt=round(dt, 4), dz_from_b_pct=round(100 * dz_b, 2),
                         e_from_b=round(e_b, 3) if e_b == e_b else "undefined (dt<=0)", dz_hole_pct=r.hole_width_pct, e_from_hole=round(e_h, 2) if e_h == e_h else ("undefined (dt<=0)" if dt <= 0 else "hole not identified")))
B = pd.DataFrame(rows); print("\nB. Reduced-form elasticities:\n" + B.to_string(index=False)); B.to_csv(f"{OUT}/rev3_elasticity.csv", index=False)

# ---------- C. sole-shareholder naming ----------
f = pd.read_csv(f"{ROOT}/data/raw/onrc/02.09.2026/OD_FIRME.CSV", sep="^", encoding="utf-8-sig", dtype=str, quoting=3, on_bad_lines="skip", index_col=False, engine="c", usecols=["DENUMIRE", "FORMA_JURIDICA"])
srl = f[f.FORMA_JURIDICA.isin(["SRL", "SRL-D"])].DENUMIRE.fillna("").str.upper()
pat = srl.str.contains("ASOCIAT UNIC|ASOCIAT-UNIC|A\\.U\\.|UNIPERSONAL", regex=True)
print(f"\nC. SRL names mentioning a sole shareholder: {pat.sum():,} of {len(srl):,} ({100*pat.mean():.2f}%); 'SRL-D' (debutant, single founder) {int((f.FORMA_JURIDICA=='SRL-D').sum()):,}")

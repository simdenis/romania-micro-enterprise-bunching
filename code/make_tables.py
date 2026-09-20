"""Generate LaTeX tables for the paper from outputs/*.csv and the panel. Run from the project root with .venv."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"; T = f"{ROOT}/paper/tables"
os.makedirs(T, exist_ok=True)
def w(name, s): open(f"{T}/{name}.tex", "w").write(s)
def tab(cols, rows, align, caption, label, note=None, size=r"\small", landscape=False, resize=True):
    s = ([r"\begin{landscape}"] if landscape else []) + [r"\begin{table}[!tbp]\centering", size, rf"\caption{{{caption}}}\label{{{label}}}", ((r"\resizebox{\linewidth}{!}{%" + "\n") if resize else r"\setlength{\tabcolsep}{3.5pt}" + "\n") + rf"\begin{{tabular}}{{{align}}}\toprule", " & ".join(cols) + r" \\ \midrule"]
    s += [" & ".join(str(x) for x in r) + r" \\" for r in rows]
    s += [r"\bottomrule\end{tabular}" + ("}" if resize else "")]
    if note: s += [rf"\begin{{minipage}}{{0.95\linewidth}}\vspace{{2pt}}\footnotesize {note}\end{{minipage}}"]
    s += [r"\end{table}"] + ([r"\end{landscape}"] if landscape else []); return "\n".join(s)
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]

# panel summary
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "turnover", "employees"]).drop_duplicates(["year", "cui"])
g = p.groupby("year").agg(firms=("cui", "size"), pos=("revenue", lambda s: (s > 0).sum()), med=("revenue", lambda s: s[s > 0].median()), emp=("employees", lambda s: s.fillna(0).median()))
rows = [(int(y), f"{int(r.firms):,}", f"{int(r.pos):,}", f"{r.med/1e3:,.0f}", f"{fx[y]:.4f}") for y, r in g.iterrows()]
w("panel", tab(["Year", "Firm-years", "Revenue $>0$", "Median revenue (k lei)", "EUR/RON, 31 Dec"], rows, "lrrrr",
    "Statement panel, 2014--2025", "tab:panel", "Source: Ministry of Finance annual financial statements (micro-entity and full forms) via data.gov.ro; BNR year-end reference rate. Median revenue is over firms with positive revenue. 32 duplicated firm-years (30 in 2016) removed; about 600 firm-years per year with negative revenue are excluded from all windows."))

# regime cross-check 2023
ct = pd.read_csv(f"{OUT}/regime_crosscheck_2023.csv", index_col=0)
order = ["micro", "micro_loss", "profit", "no_tax", "ambiguous", "other"]; lab = {"micro": "Micro (tax $\\approx$ 1--3\\% of revenue)", "micro_loss": "Micro (tax $>0$, pre-tax loss)", "profit": "Profit tax ($\\approx$ 16\\% of profit)", "no_tax": "No tax paid", "ambiguous": "Ambiguous", "other": "Other"}
rows = [(lab[r], f"{ct.loc[r,'flag_micro']:,}", f"{ct.loc[r,'flag_profit']:,}", f"{ct.loc[r,'flag_none']+ct.loc[r,'flag_both']:,}", f"{ct.loc[r,'not_in_anaf']:,}") for r in order if r in ct.index]
core = ct.loc[["micro", "micro_loss", "profit"], ["flag_micro", "flag_profit"]]
agree = (core.loc["micro", "flag_micro"] + core.loc["micro_loss", "flag_micro"] + core.loc["profit", "flag_profit"]) / core.values.sum()
w("crosscheck", tab(["Regime inferred from statements", "ANAF: micro", "ANAF: profit", "ANAF: other/none", "Not in ANAF"], rows, "lrrrr",
    "Tax regime inferred from the 2023 statements versus the ANAF tax-vector flag (June 2023)", "tab:crosscheck",
    f"ANAF flags: IMP120 = registered for micro-enterprise income tax, IMP100 = registered for profit tax. Where both measures are decisive, they agree for {agree*100:.1f}\\% of firms."))

# difference-in-bunching
d = pd.read_csv(f"{OUT}/diff_in_bunching.csv"); c = pd.read_csv(f"{OUT}/diff_in_bunching_counts.csv")
m = d.merge(c[["notch", "excess_firms_5pct", "excess_share_5pct", "excess_firms_3pct", "excess_firms_8pct"]], on="notch")
def ctrl(s):
    ys = [int(x) for x in s.split(",")]; out = []; start = prev = ys[0]
    for y in ys[1:] + [None]:
        if y is not None and y == prev + 1: prev = y; continue
        out.append(f"{start}" if start == prev else f"{start}--{prev}");
        if y is not None: start = prev = y
    return ", ".join(out)
rows = []
for _, r in m.iterrows():
    lab_ = r.notch.replace("micro €", "in-year, €").replace("next-year eligibility €", "year-end, €").replace("€", r"\euro{}")
    rows.append((lab_, f"{r.location_lei/1e6:.3f}", ctrl(r.controls), f"{r.firms_in_window:,}", f"{r.b_diff:.2f} ({r.boot_se:.2f})", f"{r.missing_mass_above:.2f}", f"{r.excess_firms_5pct:,}", f"{r.excess_share_5pct:.1f}", f"{r.excess_firms_3pct:,}--{r.excess_firms_8pct:,}"))
w("dib", tab(["Notch", "Location (M lei)", "Control years", "Firms in window", "$b$ (SE)", "$m$", "Excess firms", "\\% of window", "Range 3--8\\%"], rows, "llllrrrrr",
    "Difference-in-bunching estimates", "tab:dib",
    "Location = EUR threshold $\\times$ BNR rate at the close of the previous year (in-year notches) or of the same year (year-end notches). Window $\\pm$20\\%, bins 1\\% of the location. Control density = same-quantile density in years with no known notch within 25\\% of the matched location. $b$ = excess mass in the five bins below the notch in units of counterfactual firms per bin; $m$ = same above the notch; SE from 300 multinomial bootstrap draws. Excess firms = excess in the 5\\% region below the notch; last column gives the range for 3\\% and 8\\% regions.", r"\footnotesize"))

# lei tracking
l = pd.read_csv(f"{OUT}/lei_tracking_1M.csv")
rows = [(int(r.year), f"{int(r.thr_prev_rate):,}", f"{int(r.thr_cur_rate):,}", f"{int(r.modal_bin_center):,}", f"{int(r.dist_to_prev):+,}", f"{int(r.dist_to_cur):+,}") for _, r in l.iterrows()]
w("lei", tab(["Year", "\\euro{}1M at prev.\\ year-end rate", "\\euro{}1M at current year-end rate", "Modal bin (lei)", "Distance to prev.", "Distance to current"], rows, "lrrrrr",
    "Location of the revenue spike in lei, 2018--2022", "tab:lei", "Modal 10,000-lei bin among bins within 150,000 lei of either candidate threshold. In 2018 and 2022 the two rates nearly coincide."))

# VAT
v = pd.read_csv(f"{OUT}/vat_bunching_stats.csv")
v = pd.concat([v, pd.read_csv(f"{OUT}/rev2_vat_395k.csv").query("year == 2025")], ignore_index=True)
rows = [(int(r.year), f"{int(r.vat_threshold_lei):,}", f"{int(r.firms_in_window):,}", f"{r.excess_mass_b:.2f}") for _, r in v.iterrows()]
w("vat", tab(["Year", "VAT threshold (lei)", "Firms in window", "$b$ (polynomial)"], rows, "lrrr",
    "Bunching of net turnover at the VAT registration threshold", "tab:vat", "Polynomial counterfactual (degree 5, three bins excluded on each side), 2,000-lei bins, window $\\pm$40\\%. The threshold rose to 395,000 lei on 1 September 2025; the two 2025 rows test the old and the new location (at 395,000 lei the 2024 placebo gives 0.62)."))

# employees
e = pd.read_csv(f"{OUT}/employee_margin.csv", index_col=0); et = pd.read_csv(f"{OUT}/employee_transitions_all.csv")
rows = [(int(y), f"{r['0']:.1f}", f"{r['1']:.1f}", f"{r['2']:.1f}", f"{r['5+']:.1f}", f"{int(r.firms):,}") for y, r in e.iterrows()]
w("emp", tab(["Year", "0 employees", "1", "2", "5+", "Firms"], rows, "lrrrrr", "Employee counts among firms with revenue below \\euro{}500,000 (\\% of firms)", "tab:emp", "Average annual headcount as reported in the statements."))
rows = [(int(r.base_year), f"{int(r.firms_0emp):,}", f"{r.to_1emp/(100-r.no_filing_next)*100:.1f}", f"{r.to_2plus/(100-r.no_filing_next)*100:.1f}", f"{r.stay_0emp/(100-r.no_filing_next)*100:.1f}") for _, r in et.iterrows()]
w("emptrans", tab(["Year $t$", "Firms with 0 employees in $t$", "1 employee in $t+1$", "2+ in $t+1$", "Still 0 in $t+1$"], rows, "lrrrr",
    "Transitions of zero-employee firms (revenue below \\euro{}500,000) into the next year, conditional on filing in both years (\\%)", "tab:emptrans", "The one-employee condition was enacted in July 2022 and tested at 31 December 2022. Filing coverage differs across years because some annual files are early uploads, so shares are conditional on a statement being present in $t+1$."))

# splitting pooled
A = pd.read_csv(f"{OUT}/splitting_by_position.csv"); CZ = pd.read_csv(f"{OUT}/splitting_clustered_z.csv")
def pooled(y0, y1, col, g):
    dd = A[(A.year >= y0) & (A.year <= y1) & (A.group == g)]; n = dd.firms.sum(); return (dd[col] / 100 * dd.firms).sum() / n, n
rows = []
for col, key, name in [("pct_sib_new", "sib_new", "Administrator registered another firm, prior 18 months"), ("pct_addr_new", "addr_new", "New firm at the same address, prior 18 months"), ("pct_any_sib", "any_sib", "Administrator runs another firm, any date")]:
    vals = [pooled(2018, 2023, col, g) for g in ["bunchers (0-5% below)", "just above (0-10%)", "control (10-25% below)", "well above (25-100%)"]]
    z = CZ[(CZ.years == "2018--2023") & (CZ.measure == key)].z_clustered.item()
    rows.append((name, *[f"{v[0]*100:.2f}" for v in vals], f"{z:.1f}"))
early = {r.measure: r for _, r in CZ[CZ.years == "2015--2017"].iterrows()}
w("split", tab(["Measure", "Bunchers", "Just above", "10--25\\% below", "25--100\\% above", "$z$"], rows, "p{5.2cm}rrrrr",
    "Sibling-firm indicators by position relative to the threshold, SRLs, pooled 2018--2023", "tab:split",
    f"Pooled shares, \\%. $z$ tests bunchers against firms just above the threshold with standard errors clustered by firm. At the small ceilings of 2015--2017 the corresponding shares for bunchers and firms just above are {early['sib_new'].bunchers:.2f} versus {early['sib_new'].just_above:.2f} ($z = {early['sib_new'].z_clustered:.1f}$) on the administrator measure, {early['addr_new'].bunchers:.2f} versus {early['addr_new'].just_above:.2f} ($z = {early['addr_new'].z_clustered:.1f}$) on the address measure and {early['any_sib'].bunchers:.2f} versus {early['any_sib'].just_above:.2f} ($z = {early['any_sib'].z_clustered:.1f}$) on the any-date measure. Standardising the just-above group to the bunchers' age distribution gives 2.70 versus 2.20 on the administrator measure and 12.26 versus 10.23 on the address measure (Appendix Table~\\ref{{tab:splitage}}). Administrators as recorded in the ONRC snapshot of 2 September 2026; persons administering more than 50 firms and addresses hosting more than 20 firms excluded. Bunchers: 0--5\\% below the threshold; just above: 0--10\\% above.", r"\small", resize=False))
print("tables written:", sorted(os.listdir(T)))

# ---- revision tables ----
mg = pd.read_csv(f"{OUT}/rev_margins.csv")
pv = mg.pivot(index="year", columns="group", values="median_margin"); ps = mg.pivot(index="year", columns="group", values="share_margin_above_6pct")
rows = [(y, f"{pv.loc[y,'bunchers 0-5% below']:.3f}", f"{pv.loc[y,'just above 0-10%']:.3f}", f"{pv.loc[y,'10-25% below']:.3f}", f"{ps.loc[y,'bunchers 0-5% below']:.0f}", f"{ps.loc[y,'just above 0-10%']:.0f}") for y in pv.index]
w("margins", tab(["Year", "Bunchers (0--5\\% below)", "Just above (0--10\\%)", "10--25\\% below", "Bunchers: margin $>$ 6.25\\% (\\%)", "Just above: margin $>$ 6.25\\% (\\%)"], rows, "lrrrrr",
    "Pre-tax profit margin by position relative to the threshold", "tab:margins",
    "Median of pre-tax profit over total revenue. At a 1\\% turnover tax and a 16\\% profit tax, the regime lowers the tax bill only for firms with margins above 6.25\\%; at 3\\% the break-even margin is 18.75\\%."))
tr = pd.read_csv(f"{OUT}/rev_transition_placebo.csv").sort_values("year")
rows = [(int(r.year), "notch year" if r.year == 2023 else ("post-transition" if r.year == 2024 else "placebo"), f"{int(r.bunchers_n):,}", f"{r.bunchers_from_above:.1f}", f"{r.bunchers_from_120_200pct:.1f}", f"{int(r.controls_n):,}", f"{r.controls_from_above:.1f}", f"{r.ratio:.2f}") for _, r in tr.iterrows()]
w("transition", tab(["Year", "Status", "Firms 0--6\\% below", "\\% above \\euro{}500k in $t-1$", "\\% in 120--200\\% band in $t-1$", "Firms 14--20\\% below", "\\% above in $t-1$", "Ratio"], rows, "llrrrrrr",
    "Origin of firms just below \\euro{}500,000: share that was above \\euro{}500,000 the year before", "tab:transition",
    "Positions relative to \\euro{}500,000 at the previous year-end rate in every year, whether or not it was a threshold. Ratio = bunchers' share from above divided by the comparison group's share."))
ag = pd.read_csv(f"{OUT}/rev_splitting_by_age.csv", index_col=0)
rows = [(b, f"{int(r.n_bunchers):,}", f"{int(r.n_just_above):,}", f"{r.sib_new_bunchers:.2f}", f"{r.sib_new_just_above:.2f}", f"{r.addr_new_bunchers:.2f}", f"{r.addr_new_just_above:.2f}") for b, r in ag.iterrows()]
w("splitage", tab(["Firm age (years)", "Bunchers, $n$", "Just above, $n$", "Admin.\\ new firm: bunchers", "just above", "Same-address new firm: bunchers", "just above"], rows, "lrrrrrr",
    "Sibling-firm indicators by firm age, bunchers versus firms just above the threshold, pooled 2018--2023 (\\%)", "tab:splitage",
    "Age = year minus year of registration in the trade register. Standardising the just-above group to the bunchers' age distribution gives 2.70 versus 2.20 (administrator measure, +23\\%) and 12.26 versus 10.23 (address measure, +20\\%)."))
print("revision tables written")

# ---- second revision: robustness table, economic quantities, fine-bin lei table ----
rb = pd.read_csv(f"{OUT}/rev2_robustness.csv"); qt = pd.read_csv(f"{OUT}/rev2_quantities.csv")
m2 = rb.merge(qt[["notch", "B_over_N_pct", "b_quantile_corrected"]], on="notch")
def lab(n): return n.replace("micro €", "in-year, €").replace("announced-for-2023 €", "announced, €").replace("next-year eligibility €", "year-end, €").replace(" (end-2024 rate)", "").replace("€", r"\euro{}")
rows = [(lab(r.notch), f"{r.b_base:.2f}", f"{r.b_quantile_corrected:.2f}", f"{r.b_win10:.2f}", f"{r.b_win30:.2f}", "--" if pd.isna(r.b_pre_only) else f"{r.b_pre_only:.2f}", "--" if pd.isna(r.b_post_only) else f"{r.b_post_only:.2f}", f"{r.b_loo_min:.2f}--{r.b_loo_max:.2f}", f"{r.jackknife_se:.2f}", f"{r.b_continuing:.2f}", f"{r.B_over_N_pct:.2f}") for _, r in m2.iterrows()]
w("robust", tab(["Notch", "$b$", "quantile-corrected", "window $\\pm$10\\%", "$\\pm$30\\%", "pre-years only", "post-years only", "leave-one-out", "jackknife SE", "continuing firms", "$B/N_t$ (\\%)"], rows, "lrrrrrrrrrr",
    "Robustness of the difference-in-bunching estimates", "tab:robust",
    "Baseline: window $\\pm$20\\%, all admissible control years. Quantile-corrected: control quantile shifted down by $B/N_t$. Pre/post: only control years before/after the treated year. Leave-one-out: range of $b$ dropping one control year at a time; jackknife SE over control years. Continuing firms: firms with positive revenue in $t-1$, $t$ and $t+1$. $B/N_t$: excess firms in the 5\\% region as a share of all filers with positive revenue in year $t$.", r"\footnotesize"))
lf = pd.read_csv(f"{OUT}/rev2_lei_modal_fine.csv")
rows = [(int(r.year), f"{int(r.thr_prev):,}", r.modal_5k_bin, int(r.firms_5k), f"{int(r.dist_5k_to_prev):+,}", f"{int(r.thr_cur - r.thr_prev):+,}") for _, r in lf.iterrows()]
w("lei", tab(["Year", "\\euro{}1M, prev.\\ rate (lei)", "Modal 5,000-lei bin", "Firms", "Bin centre $-$ threshold", "Current $-$ prev.\\ rate"], rows, "lrlrrr",
    "Location of the revenue spike in lei, 2018--2022", "tab:lei", "Modal 5,000-lei bin within 60,000 lei of either candidate threshold. The last column gives the lei value of \\euro{}1M at the current year-end rate minus its value at the previous year-end rate (the dotted line in Figure~\\ref{fig:lei}). At 1,000-lei resolution the modal bins hold only 16 to 31 firms and are noisier; they lie 2,000 to 5,000 lei below the previous-rate threshold in 2020--2022 and further below in 2018--2019.", r"\footnotesize", resize=False))
print("second-revision tables written")

# ---- third revision: block bootstrap column, elasticity appendix table ----
bb = pd.read_csv(f"{OUT}/rev3_block_bootstrap.csv"); rb = pd.read_csv(f"{OUT}/rev2_robustness.csv"); qt = pd.read_csv(f"{OUT}/rev2_quantities.csv")
m3 = rb.merge(qt[["notch", "B_over_N_pct", "b_quantile_corrected"]], on="notch").merge(bb[["notch", "block_boot_se"]], on="notch")
rows = [(lab(r.notch), f"{r.b_base:.2f}", f"{r.block_boot_se:.2f}", f"{r.b_win10:.2f}", f"{r.b_win30:.2f}", f"{r.b_loo_min:.2f}--{r.b_loo_max:.2f}", f"{r.B_over_N_pct:.2f}") for _, r in m3.iterrows()]
w("robust", tab(["Notch", "$b$", "Block-bootstrap SE", "Window $\\pm$10\\%", "Window $\\pm$30\\%", "Leave-one-out range", "$B/N_t$ (\\%)"], rows, "lrrrrrr",
    "Robustness of the difference-in-bunching estimates", "tab:robust",
    "Baseline: window $\\pm$20\\%, all admissible control years. Block-bootstrap SE: control years resampled with replacement and treated bin counts resampled, 500 draws. Leave-one-out: range of $b$ dropping one control year at a time. $B/N_t$: excess firms in the 5\\% region as a share of all filers with positive revenue in year $t$. Further checks in Appendix Table~\\ref{tab:robustextra}.", r"\small"))
rows = [(lab(r.notch), f"{r.b_base:.2f}", f"{r.b_quantile_corrected:.2f}", "--" if pd.isna(r.b_pre_only) else f"{r.b_pre_only:.2f}", "--" if pd.isna(r.b_post_only) else f"{r.b_post_only:.2f}", f"{r.b_continuing:.2f}", f"{int(r.n_continuing):,}", f"{r.jackknife_se:.2f}") for _, r in m3.iterrows()]
w("robustextra", tab(["Notch", "$b$", "Quantile-corrected", "Pre-years only", "Post-years only", "Continuing firms", "$n$ continuing", "Jackknife SE"], rows, "lrrrrrrr",
    "Further robustness checks of the difference-in-bunching estimates", "tab:robustextra",
    "Quantile-corrected: control quantile shifted down by $B/N_t$. Pre/post: only control years before/after the treated year. Continuing: firms with positive revenue in $t-1$, $t$ and $t+1$. Jackknife SE over control years.", r"\small"))
el = pd.read_csv(f"{OUT}/rev3_elasticity.csv"); el = el[el.margin_basis == "median buncher margin"]
def fmt(x): return x if isinstance(x, str) else f"{x:.3f}"
rows = [(lab(r.notch), f"{r.tau*100:.0f}\\%", f"{r.margin:.3f}", f"{r['dt']*100:+.2f}", f"{r.dz_from_b_pct:.2f}", fmt(r.e_from_b), "--" if pd.isna(r.dz_hole_pct) else f"{r.dz_hole_pct:.0f}", fmt(r.e_from_hole) if isinstance(r.e_from_hole, str) else f"{float(r.e_from_hole):.2f}") for _, r in el.iterrows()]
w("elasticity", tab(["Notch", "$\\tau$", "median buncher margin", "$\\Delta t$ (pp)", "$\\Delta z^*/z^*$ from $b$ (\\%)", "$e$ from $b$", "hole width (\\%)", "$e$ from hole"], rows, "lrrrrrrr",
    "Reduced-form elasticities under the Kleven--Waseem approximation, reported for completeness", "tab:elasticity",
    "$\\Delta t = 0.16 \\times \\text{margin} - \\tau$ is the change in the average tax rate on revenue at the notch for a firm with the median reported margin of bunchers; $e = (\\Delta z^*/z^*)^2 / (2\\Delta t/(1-\\tau))$. Two proxies for $\\Delta z^*/z^*$: $b$ with 1\\% bins, and the width above the notch at which the cumulative deficit offsets the excess (the hole), which is not reached before the window edge in most years. The proxies differ by three orders of magnitude and $\\Delta t \\le 0$ in the 3\\% years, which is why the text does not rely on these numbers.", r"\footnotesize"))
print("third-revision tables written")

# ---- fourth revision: event study, group bunching, sector tables; split Table 1 handled in main.tex; narrower dib table ----
ev = pd.read_csv(f"{OUT}/rev4_margin_event_study.csv")
rows = []
for des, lab_ in [("2018: (500k,1M] moved to micro vs (1M,2M]", "2018 increase: (500k, 1M] vs (1M, 2M] on 2017 revenue"), ("2023: (500k,1M] lost micro vs (1M,2M]", "2023 cut: (500k, 1M] vs (1M, 2M] on 2022 revenue"), ("placebo: bands on 2019 revenue", "Placebo: same bands on 2019 revenue, no regime change")]:
    d = ev[ev.design == des]; rows.append((r"\multicolumn{7}{l}{\emph{" + lab_ + "}}",))
    for _, r in d.iterrows():
        rows.append((int(r.year), f"{r.micro_share_T:.0f} / {r.micro_share_C:.0f}", f"{100*r.median_margin_T:.2f}", f"{100*r.median_margin_C:.2f}", f"{100*r.DiD_median_margin:+.2f}", f"{100*r.DiD_mean_margin:+.2f}", f"{int(r.median_rev_T_keur):,} / {int(r.median_rev_C_keur):,}"))
w("eventstudy", tab(["Year", "Micro share T / C (\\%)", "Median margin T (\\%)", "Median margin C (\\%)", "DiD median (pp)", "DiD mean (pp)", "Median revenue T / C (k\\euro{})"], rows, "lllllll",
    "Reported pre-tax margins around regime switches induced by ceiling changes", "tab:eventstudy",
    "Balanced panels of firms with positive revenue in every year shown. T = firms whose base-year revenue placed them in the band that changed regime; C = firms in the band above that did not. DiD = change in T minus change in C relative to the base year (2017, 2022, 2019). Margins are pre-tax profit over revenue, winsorised at $\\pm$1. The placebo shows the drift induced by selecting on base-year revenue alone.", r"\footnotesize"))
gb = pd.read_csv(f"{OUT}/rev4_group_bunching.csv")
rows = [(int(r.year), f"{int(r.groups):,}", f"{int(r.groups_in_window):,}", f"{r.b_group_total:.2f}", f"{int(r.excess_groups)}", f"{r.firm_level_bunch_share_in_groups:.1f}", f"{r.firm_level_bunch_share_singletons:.1f}") for _, r in gb.iterrows()]
w("groups", tab(["Year", "Groups (2--5 SRLs, one administrator)", "Groups in window", "$b$, combined revenue at \\euro{}500k", "Excess groups", "Firms in groups: \\% in 5\\% below own notch", "Single firms: same"], rows, "lrrrrrr",
    "Bunching of combined revenue of firms sharing an administrator, and firm-level bunching inside groups", "tab:groups",
    "From 2024 the ceiling is tested on the combined revenue of linked enterprises. Combined revenue of each administrator's firms, window $\\pm$20\\% around \\euro{}500,000 at the previous year-end rate, controls = same-quantile density of combined revenue in 2019--2022 excluding years where the individual \\euro{}1M notch is within 25\\%. Last two columns: share of firms within 20\\% of that year's individual notch that sit in the 5\\% just below it.", r"\footnotesize", landscape=True))
sc_ = pd.read_csv(f"{OUT}/rev4_sector.csv")
piv = sc_.pivot(index="name", columns="year", values="excess_share_pct"); pm = sc_.pivot(index="name", columns="year", values="median_margin_bunchers"); pn = sc_.pivot(index="name", columns="year", values="firms_in_window")
order = piv[2023].sort_values(ascending=False).index
rows = [(n, *[("--" if pd.isna(piv.loc[n, y]) else f"{piv.loc[n, y]:.1f}") for y in (2021, 2023, 2025)], "--" if pd.isna(pn.loc[n, 2023]) else f"{int(pn.loc[n, 2023]):,}", "--" if pd.isna(pm.loc[n, 2023]) else f"{pm.loc[n, 2023]:.2f}") for n in order]
w("sector", tab(["CAEN section", "2021 (\\euro{}1M)", "2023 (\\euro{}500k)", "2025 (\\euro{}250k)", "Firms in 2023 window", "Median margin of 2023 bunchers"], rows, "lrrrrr",
    "Excess mass by sector, in percent of firms within 20\\% of the notch", "tab:sector",
    "Difference-in-bunching within each CAEN section, matched at the section's own quantile, same control years as the pooled estimate; sections with fewer than 800 firms in the window omitted. Hospitality faced no revenue ceiling in 2023."))
# narrower main dib table: move control years to an appendix table
d = pd.read_csv(f"{OUT}/diff_in_bunching.csv"); c = pd.read_csv(f"{OUT}/diff_in_bunching_counts.csv"); m = d.merge(c[["notch", "excess_firms_5pct", "excess_share_5pct", "excess_firms_3pct", "excess_firms_8pct"]], on="notch")
rows = [(lab(r.notch), f"{r.location_lei/1e6:.3f}", f"{r.firms_in_window:,}", f"{r.b_diff:.2f} ({r.boot_se:.2f})", f"{r.missing_mass_above:.2f}", f"{r.excess_firms_5pct:,}", f"{r.excess_share_5pct:.1f}", f"{r.excess_firms_3pct:,}--{r.excess_firms_8pct:,}") for _, r in m.iterrows()]
w("dib", tab(["Notch", "Location (M lei)", "Firms in window", "$b$ (SE)", "$m$", "Excess firms", "\\% of window", "Range 3--8\\%"], rows, "llrrrrrr",
    "Difference-in-bunching estimates", "tab:dib",
    "Location = EUR threshold $\\times$ BNR rate at the close of the previous year (in-year notches; the 2025 year-end notch at the end-2024 rate). Window $\\pm$20\\%, bins 1\\% of the location. Control years are listed in Appendix Table~\\ref{tab:controls}. $b$ = excess mass in the five bins below the notch in units of counterfactual firms per bin; $m$ = the same over the five bins above, negative meaning a deficit (a hole) and positive an excess; SE from 300 multinomial bootstrap draws. Excess firms = excess in the 5\\% region below the notch, with its range over 3\\% and 8\\% regions.", r"\footnotesize"))
rows = [(lab(r.notch), ctrl(r.controls), f"{r['quantile']:.3f}") for _, r in d.iterrows()]
w("controls", tab(["Notch", "Admissible control years", "Quantile of notch"], rows, "lll", "Control years used for each notch", "tab:controls", "A year is admissible if no known notch lies within 25\\% of the revenue at the same quantile in that year."))
print("fourth-revision tables written")

# ---- fifth revision ----
# narrowed event study
ev = pd.read_csv(f"{OUT}/rev5_event_study_narrow.csv"); ev = ev[ev.stat == "median"]
rows = []
for lab_, key in (("2018 entry: two-year average revenue in (\\euro{}0.8M, \\euro{}1M] vs (\\euro{}1M, \\euro{}1.2M], base 2017", "2018 entry"), ("2023 exit: same bands on 2021--2022 average, base 2022", "2023 exit"), ("Placebo: same bands on 2018--2019 average, base 2019", "placebo 2019")):
    rows.append((r"\multicolumn{5}{l}{\emph{" + lab_ + "}}",))
    for _, r in ev[ev.event == key].iterrows():
        rows.append((int(r.year), f"{int(r.horizon):+d}", f"{r.DiD_pp:+.2f}", "--" if pd.isna(r.placebo_same_horizon_pp) or key == "placebo 2019" else f"{r.placebo_same_horizon_pp:+.2f}", "--" if pd.isna(r.corrected_pp) else (f"{r.corrected_pp:+.2f}" + (f" ({r.corrected_se_pp:.2f})" if not pd.isna(r.corrected_se_pp) else ""))))
w("eventnarrow", tab(["Year", "Horizon", "DiD, median margin (pp)", "Placebo DiD, same horizon", "Placebo-corrected (SE)"], rows, "lllll",
    "Reported margins around regime switches, narrowed bands", "tab:eventnarrow",
    "Balanced panels; bands assigned on the average of revenue in the base year and the year before, within 20\\% of the \\euro{}1,000,000 cut-off. DiD = change in the treated band's median pre-tax margin minus the change in the control band's, relative to the base year. Placebo-corrected = DiD minus the placebo DiD at the same horizon; SE from 200 bootstrap draws over firms. Treated/control firms: 5,353/3,760 (2018), 7,556/4,766 (2023), 6,222/3,940 (placebo).", r"\footnotesize"))
# control locations
cl = pd.read_csv(f"{OUT}/rev5_control_locations.csv"); cl = cl[cl.admissible]
rows = []
for notch, g in cl.groupby("notch", sort=False):
    cells = "; ".join(f"{int(r.control_year)}: {r.matched_revenue_lei/1e6:.2f}M ({r.distance_pct:.0f}\\%)" for _, r in g.iterrows())
    rows.append((lab(notch), cells))
w("controls", tab(["Notch", "Control year: matched revenue $L^*_c$ (distance to nearest notch in that year)"], rows, "lp{11.5cm}",
    "Admissible control years, matched locations and distance to the nearest notch", "tab:controls",
    "$L^*_c$ = revenue at the notch's quantile in the control year. Distance = $|n/L^*_c - 1|$ for the nearest known notch $n$ of that year (micro ceiling, future ceiling, \\euro{}60,000 boundary or VAT threshold); admissible if above 25\\%. The closest cases are 26--34\\%.", r"\footnotesize"))
# 60k boundary
b60 = pd.read_csv(f"{OUT}/rev5_60k_boundary.csv")
rows = [(int(r.year), f"{int(r.eur60k_lei):,}", "300,000 (to Aug.\\ 2025)" if r.year == 2025 else "300,000", f"{r.b_at_60k:.2f}", f"{r.b_at_300k:.2f}", r.modal_2k_bin_250_450k, f"{int(r.modal_count):,} / {int(r.median_count):,}") for _, r in b60.iterrows()]
w("boundary60k", tab(["Year", "\\euro{}60,000 in lei", "VAT threshold (lei)", "$b$ at \\euro{}60,000", "$b$ at 300,000", "Modal 2,000-lei bin, 250--450k", "Firms in modal / median bin"], rows, "lrrrrlr",
    "The 1\\%/3\\% rate boundary at \\euro{}60,000 and the VAT threshold, 2023--2025", "tab:boundary60k",
    "Polynomial excess mass, 1,500-lei bins, window $\\pm$30\\%. In 2023 no rate boundary existed; in 2024 the boundary and the VAT threshold lie 1,500 lei apart; in 2025 the VAT threshold moved to 395,000 lei on 1 September."))
print("fifth-revision tables written")


# ---- sixth revision: contemporaneous placebos ----
ec = pd.read_csv(f"{OUT}/rev6_event_study_contemp.csv")
rows = []
for lab_, key in (("2018 entry: bands on 2016--2017 average revenue, cut-off \\euro{}1M; placebos at \\euro{}1.5M and \\euro{}2M", "2018 entry"), ("2023 exit: bands on 2021--2022 average revenue, cut-off \\euro{}1M; placebos at \\euro{}1.5M and \\euro{}2M", "2023 exit")):
    rows.append((r"\multicolumn{6}{l}{\emph{" + lab_ + "}}",))
    for _, r in ec[ec.event == key].iterrows():
        f = lambda v, se: ("0.00" if r.horizon == 0 else f"{v:+.2f} ({se:.2f})")
        rows.append((int(r.year), f"{int(r.horizon):+d}", f(r.raw_DiD_pp, r.raw_se_pp), f"{r.placebo_1_5M_pp:+.2f}", f"{r.placebo_2M_pp:+.2f}", f(r.corrected_pp, r.corrected_se_pp)))
w("eventnarrow", tab(["Year", "Horizon", "DiD, median margin, pp (SE)", "Placebo at \\euro{}1.5M", "Placebo at \\euro{}2M", "Corrected, pp (SE)"], rows, "llllll",
    "Reported margins around regime switches: narrowed bands with contemporaneous placebos", "tab:eventnarrow",
    "Balanced panels. Treated band: two-year average revenue in (0.8, 1.0] $\\times$ the cut-off; control band: (1.0, 1.2] $\\times$ the cut-off; DiD = change in the treated band's median pre-tax margin minus the change in the control band's, relative to the base year. Placebos: the same construction at cut-offs of \\euro{}1.5M and \\euro{}2M in the same base year, where no regime changed. Corrected = DiD minus the mean of the two placebos. SE from 200 bootstrap draws over firms. Firms: 5,353/3,760 treated/control in 2018, 7,556/4,766 in 2023; placebo designs 2,200--4,800 per band.", r"\footnotesize"))
print("sixth-revision table written")

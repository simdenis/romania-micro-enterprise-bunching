"""Firm-splitting test using the ONRC administrator linkage.

Question: when a firm is near the micro threshold, does its administrator register another firm?
Measures for firm i in year t:
  sib_new_t : an administrator of i also administers another firm registered between 1 Jul t-1 and 31 Dec t
  n_sib     : number of other firms the administrators of i run (any date)
  addr_new_t: another firm registered at the same address between 1 Jul t-1 and 31 Dec t
Groups by position relative to the year's binding threshold (previous year-end rate):
  bunchers 0-5% below | control 10-25% below | just above 0-10% | well above 25-100% above.
Two designs: (A) cross-section by position, notch years vs placebo locations; (B) 2022 cohort about to lose
micro status in 2023 (revenue 500k-1M EUR in 2022) vs unaffected 250k-500k, registrations of sibling firms in H2 2022-2023,
with 2019 and 2020 as placebo cohorts.
Caveat: administrators are as of the Sept-2026 snapshot (current, not historical); shareholders are not in the open data.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"; PAN = f"{ROOT}/data/panel"
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
MICRO = {2014: 65e3, 2015: 65e3, 2016: 100e3, 2017: 500e3, 2018: 1e6, 2019: 1e6, 2020: 1e6, 2021: 1e6, 2022: 1e6, 2023: 500e3, 2024: 500e3, 2025: 250e3}

p = pd.read_parquet(f"{PAN}/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "employees"]).drop_duplicates(["year", "cui"])
firms = pd.read_parquet(f"{PAN}/onrc_firms.parquet")
link = pd.read_parquet(f"{PAN}/onrc_person_firm.parquet")          # person_id, cui, reg_date
srl = set(firms.loc[firms.FORMA_JURIDICA.isin(["SRL", "SRL-D"]), "cui"])
linked = set(link.cui.dropna())
p = p[p.cui.isin(srl) & p.cui.isin(linked) & (p.revenue > 0)].copy()   # firms with >=1 linked administrator only
p["rev_eur"] = p.revenue / p.year.map(lambda y: fx[y - 1])
p["rel"] = p.rev_eur / p.year.map(MICRO) - 1

# person -> all firms with registration dates (integer keys, multi-firm persons only, nominees with >50 firms dropped)
pf = link[["person_id", "code", "cui", "reg_date"]].dropna(subset=["reg_date"]).copy()
pf["pid"] = pd.factorize(pf.person_id)[0]; pf["fid"] = pd.factorize(pf.code)[0]
npp = pf.groupby("pid").fid.transform("size"); pf = pf[(npp >= 2) & (npp <= 50)][["pid", "fid", "cui", "reg_date"]]
sib = pf.merge(pf[["pid", "fid", "reg_date"]], on="pid", suffixes=("", "_sib"))
sib = sib[(sib.fid != sib.fid_sib) & sib.cui.notna()][["cui", "fid_sib", "reg_date_sib"]].drop_duplicates()
n_sib = sib.groupby("cui").fid_sib.nunique().rename("n_sib")
# address siblings: addresses shared by 2-20 firms only
fa = firms[["cui", "addr_id", "reg_date"]].dropna().copy()
na = fa.groupby("addr_id").cui.transform("size"); fa = fa[(na >= 2) & (na <= 20)]
fa["aid"] = pd.factorize(fa.addr_id)[0]; fa = fa[["cui", "aid", "reg_date"]]
asib = fa.merge(fa, on="aid", suffixes=("", "_sib")); asib = asib[asib.cui != asib.cui_sib][["cui", "cui_sib", "reg_date_sib"]]
del pf, fa

def flag_window(pairs, year):
    lo, hi = pd.Timestamp(year - 1, 7, 1), pd.Timestamp(year, 12, 31)
    return set(pairs.loc[(pairs.reg_date_sib >= lo) & (pairs.reg_date_sib <= hi), "cui"])

rows = []
for y in range(2015, 2026):
    d = p[p.year == y].copy()
    d["n_sib"] = d.cui.map(n_sib).fillna(0)
    s_new, a_new = flag_window(sib, y), flag_window(asib, y)
    d["sib_new"] = d.cui.isin(s_new); d["addr_new"] = d.cui.isin(a_new)
    groups = {"bunchers (0-5% below)": (d.rel >= -0.05) & (d.rel < 0), "control (10-25% below)": (d.rel >= -0.25) & (d.rel < -0.10),
              "just above (0-10%)": (d.rel >= 0) & (d.rel < 0.10), "well above (25-100%)": (d.rel >= 0.25) & (d.rel < 1.0)}
    for g, m in groups.items():
        rows.append(dict(year=y, threshold=int(MICRO[y]), group=g, firms=int(m.sum()), pct_sib_new=round(100 * d.loc[m, "sib_new"].mean(), 2),
                         pct_any_sib=round(100 * (d.loc[m, "n_sib"] > 0).mean(), 2), mean_n_sib=round(d.loc[m, "n_sib"].mean(), 3),
                         pct_addr_new=round(100 * d.loc[m, "addr_new"].mean(), 2)))
A = pd.DataFrame(rows); pd.set_option("display.width", 250)
print("A. Share of SRLs whose administrator registered another firm between Jul(t-1) and Dec(t), by position vs threshold:")
print(A.pivot(index="year", columns="group", values="pct_sib_new").to_string())
print("\n   Share with any sibling firm (same administrator, any date):")
print(A.pivot(index="year", columns="group", values="pct_any_sib").to_string())
print("\n   Share with another firm registered at the same address in the window:")
print(A.pivot(index="year", columns="group", values="pct_addr_new").to_string())
A.to_csv(f"{OUT}/splitting_by_position.csv", index=False)

# B. cohorts about to lose status
rows = []
for y, lo, hi in [(2022, 500e3, 1e6), (2019, 500e3, 1e6), (2020, 500e3, 1e6), (2024, 250e3, 500e3), (2021, 250e3, 500e3)]:
    d = p[(p.year == y)]
    for name, m in [("affected band", (d.rev_eur >= lo) & (d.rev_eur < hi)), ("unaffected band (half the size)", (d.rev_eur >= lo / 2) & (d.rev_eur < lo))]:
        ids = d.loc[m, "cui"]
        w_lo, w_hi = pd.Timestamp(y, 7, 1), pd.Timestamp(y + 1, 12, 31)
        s = sib[sib.cui.isin(ids) & (sib.reg_date_sib >= w_lo) & (sib.reg_date_sib <= w_hi)].cui.nunique()
        a = asib[asib.cui.isin(ids) & (asib.reg_date_sib >= w_lo) & (asib.reg_date_sib <= w_hi)].cui.nunique()
        rows.append(dict(cohort_year=y, band=name, eur_from=int(lo if name.startswith("aff") else lo / 2), eur_to=int(hi if name.startswith("aff") else lo),
                         firms=len(ids), pct_admin_new_firm_next18m=round(100 * s / len(ids), 2), pct_same_addr_new_firm=round(100 * a / len(ids), 2)))
B = pd.DataFrame(rows)
print("\nB. Cohorts: share whose administrator registered a new firm between Jul(t) and Dec(t+1). 2022 cohort 500k-1M lost micro status in 2023;\n   2024 cohort 250k-500k lost it in 2025; other years are placebos:")
print(B.to_string(index=False)); B.to_csv(f"{OUT}/splitting_cohorts.csv", index=False)

"""Contemporaneous placebos for the narrowed event study: same bands (two-year average revenue within 20% of a cut-off),
same base years (2017, 2022) and horizons, at cut-offs of EUR 1.5M and 2M where no regime change occurred."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/outputs"
fx = pd.read_csv(f"{ROOT}/data/bnr_eur_yearend.csv").set_index("year")["eur_ron"]
pp = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui", "revenue", "pretax"])
pp["margin"] = (pp.pretax / pp.revenue).clip(-1, 1); pp["rev_eur"] = pp.revenue / pp.year.map(lambda y: fx[y])
rng = np.random.default_rng(9)
def build(base, years, cut, band=0.20):
    d = pp[pp.year.isin(years) & (pp.revenue > 0)]; cnt = d.groupby("cui").year.nunique(); keep = set(cnt[cnt == len(years)].index)
    a = pp[pp.year.isin([base - 1, base]) & pp.cui.isin(keep)].groupby("cui").rev_eur.mean()
    T = set(a[(a > (1 - band) * cut) & (a <= cut)].index); C = set(a[(a > cut) & (a <= (1 + band) * cut)].index)
    w = d[d.cui.isin(T | C)].pivot(index="cui", columns="year", values="margin"); w["T"] = w.index.isin(T); return w
def did(w, base, y): return (w.loc[w["T"], y].median() - w.loc[w["T"], base].median()) - (w.loc[~w["T"], y].median() - w.loc[~w["T"], base].median())
rows = []
for lab, base, years in (("2018 entry", 2017, list(range(2015, 2021))), ("2023 exit", 2022, list(range(2020, 2026)))):
    W = build(base, years, 1e6); P15 = build(base, years, 1.5e6); P20 = build(base, years, 2e6)
    print(f"{lab}: treated design T={W['T'].sum():,} C={(~W['T']).sum():,}; placebo 1.5M T={P15['T'].sum():,} C={(~P15['T']).sum():,}; placebo 2M T={P20['T'].sum():,} C={(~P20['T']).sum():,}")
    for y in years:
        raw, p15, p20 = did(W, base, y), did(P15, base, y), did(P20, base, y); corr = raw - 0.5 * (p15 + p20)
        bs_raw, bs_corr = [], []
        if y != base:
            for _ in range(200):
                s = lambda X: X.sample(len(X), replace=True, random_state=rng.integers(1e9))
                r_ = did(s(W), base, y); c_ = r_ - 0.5 * (did(s(P15), base, y) + did(s(P20), base, y)); bs_raw.append(r_); bs_corr.append(c_)
        rows.append(dict(event=lab, year=y, horizon=y - base, raw_DiD_pp=round(100 * raw, 2), raw_se_pp=round(100 * np.std(bs_raw), 2) if bs_raw else np.nan,
                         placebo_1_5M_pp=round(100 * p15, 2), placebo_2M_pp=round(100 * p20, 2), corrected_pp=round(100 * corr, 2), corrected_se_pp=round(100 * np.std(bs_corr), 2) if bs_corr else np.nan))
R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.to_string(index=False)); R.to_csv(f"{OUT}/rev6_event_study_contemp.csv", index=False)

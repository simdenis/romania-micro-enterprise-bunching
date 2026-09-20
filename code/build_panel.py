"""Build a harmonized firm-year panel from the Finance Ministry statement files in data/raw/fin/.

Each year has two files: UU (simplified / micro-entity form) and BL (long and short forms).
Column layouts differ slightly before 2015, so each (year, form) is mapped explicitly.
Output: data/panel/statements_2014_2025.parquet with one row per firm-year.
"""
import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/data/raw/fin"
OUT = f"{ROOT}/data/panel"
os.makedirs(OUT, exist_ok=True)

# Standard 20-indicator layout used from 2015 on (and by 2014 UU-2019 re-issue). Names per the .csv dictionaries.
STD20 = ["fixed_assets", "current_assets", "inventories", "receivables", "cash", "prepaid_exp",
         "liabilities", "deferred_income", "provisions", "equity", "share_capital", "patrimony_regie",
         "turnover", "revenue", "costs", "profit_gross", "loss_gross", "profit_net", "loss_net", "employees"]
# 2014 long/short form: extra "patrimoniul public" after patrimoniul regiei (21 indicators)
BL21 = STD20[:12] + ["patrimony_public"] + STD20[12:]
# 2014 simplified form: 19 indicators (no patrimoniul regiei; "capital" instead of "capital subscris varsat")
UU19 = STD20[:10] + ["share_capital"] + STD20[12:]

def layout(ncols):
    n = ncols - 2  # minus CUI, CAEN
    return {20: STD20, 21: BL21, 19: UU19}[n]

frames = []
for fn in sorted(os.listdir(RAW)):
    if not fn.endswith(".txt"):
        continue
    year, form = int(fn[:4]), fn[5:7]
    path = f"{RAW}/{fn}"
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip().split(",")
    names = ["cui", "caen"] + layout(len(header))
    df = pd.read_csv(path, sep=",", header=0, names=names, dtype=str, encoding="utf-8",
                     encoding_errors="replace", on_bad_lines="warn", low_memory=False)
    df["cui"] = pd.to_numeric(df["cui"], errors="coerce")
    df = df.dropna(subset=["cui"])
    df["cui"] = df["cui"].astype("int64")
    df["caen"] = pd.to_numeric(df["caen"], errors="coerce").astype("Int64")
    for c in names[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["patrimony_regie", "patrimony_public"]:
        if c not in df:
            df[c] = np.nan
    df["year"] = year
    df["form"] = form
    frames.append(df[["year", "form", "cui", "caen"] + STD20 + ["patrimony_public"]])
    print(f"{fn}: {len(df):>8,} rows, {len(header)} cols, dup CUI within file: {df['cui'].duplicated().sum():,}", flush=True)

panel = pd.concat(frames, ignore_index=True).drop_duplicates(["year", "cui"], keep="first")   # 32 exact duplicate firm-years, mostly 2016
# derived quantities (lei)
panel["pretax"] = panel["profit_gross"].fillna(0) - panel["loss_gross"].fillna(0)
panel["net"] = panel["profit_net"].fillna(0) - panel["loss_net"].fillna(0)
panel["tax"] = panel["pretax"] - panel["net"]
dups = panel.duplicated(["year", "cui"]).sum()
print(f"\npanel: {len(panel):,} firm-years, {panel['cui'].nunique():,} firms, duplicate (year,cui): {dups:,}")
print(panel.groupby("year").agg(firms=("cui", "size"), with_turnover=("turnover", lambda s: (s > 0).sum()),
                                 median_turnover=("turnover", "median")).to_string())
panel.to_parquet(f"{OUT}/statements_2014_2025.parquet", index=False)
print("saved", f"{OUT}/statements_2014_2025.parquet")

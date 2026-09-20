"""Crosswalk old ONRC codes (J40/123/2019) to the new 14-char codes (J2019000123 40x): new = J + YYYY + seq(6) + county(2) + one
extra digit. For each ANAF old code the ten candidates are checked against the set of codes present in the ONRC snapshot.
Also assigns every ONRC code an approximate registration date (year from the code, month from the rank of the sequence
number within county-year), so sibling firms not in ANAF (registered after June 2024) can still be dated."""
import os, hashlib, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/data/panel"
adm = pd.read_parquet(f"{OUT}/onrc_admins.parquet"); firms = pd.read_parquet(f"{OUT}/onrc_firms.parquet")
st = pd.read_csv(f"{ROOT}/data/raw/onrc/02.09.2026/OD_STARE_FIRMA.CSV", sep="^", encoding="utf-8-sig", dtype=str, quoting=3, on_bad_lines="skip", index_col=False)
st.columns = [c.strip() for c in st.columns]
allcodes = pd.Series(pd.concat([adm.COD_INMATRICULARE, st.COD_INMATRICULARE]).unique()); codeset = set(allcodes)
# --- parse every ONRC code into (prefix, year, seq, county)
o = allcodes.str.extract(r"^([JCF])(\d+)/(\d+)/(\d{4})$"); n = allcodes.str.extract(r"^([JCF])(\d{4})(\d{6})(\d{2})(\d)$")
parsed = pd.DataFrame({"code": allcodes, "pfx": o[0].fillna(n[0]), "yr": o[3].fillna(n[1]), "seq": o[2].fillna(n[2]), "cty": o[1].fillna(n[3]), "newfmt": n[0].notna()})
parsed = parsed.dropna(subset=["yr", "seq", "cty"]); parsed["yr"] = parsed.yr.astype(int); parsed["seq"] = parsed.seq.astype(int); parsed["cty"] = parsed.cty.astype(int)
parsed["rank"] = parsed.groupby(["cty", "yr"]).seq.rank(pct=True)
parsed["reg_date_code"] = pd.to_datetime(dict(year=parsed.yr, month=np.clip(np.ceil(parsed["rank"] * 12), 1, 12).astype(int), day=15), errors="coerce")
print(f"ONRC codes parsed: {len(parsed):,} (new format {parsed.newfmt.mean()*100:.1f}%)")
# --- crosswalk ANAF old codes -> new codes
a = firms.COD_INMATRICULARE.str.extract(r"^([JCF])(\d+)/(\d+)/(\d{4})$").dropna(); a.columns = ["pfx", "cty", "seq", "yr"]
a["cui"] = firms.loc[a.index, "cui"].values; a["old"] = firms.loc[a.index, "COD_INMATRICULARE"].values
base = a.pfx + a.yr + a.seq.astype(int).map("{:06d}".format) + a.cty.astype(int).map("{:02d}".format)
found = pd.Series([None] * len(a), index=a.index, dtype=object)
for x in range(10):
    cand = base + str(x); hit = cand.isin(codeset) & found.isna(); found[hit] = cand[hit]
a["new"] = found
in_old = a.old.isin(codeset)
print(f"ANAF firms: {len(a):,}; old code present in ONRC: {in_old.mean()*100:.1f}%; new code found: {a.new.notna().mean()*100:.1f}%; either: {(in_old | a.new.notna()).mean()*100:.1f}%")
both = in_old & a.new.notna(); print(f"  both present (should be rare): {both.mean()*100:.2f}%")
recent = a[a.yr.astype(int) >= 2023]; print(f"  firms registered 2023+: either present {((recent.old.isin(codeset)) | recent.new.notna()).mean()*100:.1f}%")
# code -> cui map (both spellings)
c2c = pd.concat([pd.DataFrame({"code": a.old, "cui": a.cui}), pd.DataFrame({"code": a.new.dropna(), "cui": a.loc[a.new.notna(), "cui"]})]).drop_duplicates("code")
# --- person-firm table over ALL ONRC firms: cui where known, date from ANAF when known else from the code
link = adm[["COD_INMATRICULARE", "person_id"]].rename(columns={"COD_INMATRICULARE": "code"}).merge(c2c, on="code", how="left")
link = link.merge(parsed[["code", "reg_date_code"]], on="code", how="left").merge(firms[["cui", "reg_date"]], on="cui", how="left")
link["reg_date"] = link.reg_date.fillna(link.reg_date_code)
link = link[["person_id", "code", "cui", "reg_date"]].drop_duplicates(["person_id", "code"])
link.to_parquet(f"{OUT}/onrc_person_firm.parquet", index=False)
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui"]); lc = set(link.cui.dropna())
print(f"person-firm rows: {len(link):,}; with CUI: {link.cui.notna().mean()*100:.1f}%; statement filers with >=1 linked administrator:",
      p.groupby("year").cui.apply(lambda s: round(100 * s.isin(lc).mean(), 1)).loc[[2019, 2022, 2023, 2024, 2025]].to_dict())

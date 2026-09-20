"""Stage 2 (fallback): build the firm table from the ANAF June-2024 taxpayer registry instead of ONRC OD_FIRME.
Registration code = JUDET_COMERT/NR_COMERT/AN_COMERT. Incorporation date is approximated as year AN_COMERT and a month
from the rank of NR_COMERT within (county, year) (numbers are sequential within a county-year)."""
import os, hashlib, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/data/panel"
cols = ["COD_FISCAL", "DENUMIRE", "TIP_CONTRIB", "LOCALITATE", "STRADA", "NR", "JUDET_COMERT", "NR_COMERT", "AN_COMERT", "JUDET", "AP", "DATA_RADIERE"]
a = pd.read_csv(f"{ROOT}/data/raw/anaf/anaf2024/date_identificare_platitori_2024.csv", sep="|", encoding="utf-8-sig", dtype=str,
                usecols=cols, quoting=3, on_bad_lines="skip", index_col=False, engine="c")
a = a[a.TIP_CONTRIB.str.strip() == "PJ"].copy()
a["cui"] = pd.to_numeric(a.COD_FISCAL, errors="coerce"); a = a.dropna(subset=["cui"]); a["cui"] = a.cui.astype("int64")
for c in ["JUDET_COMERT", "NR_COMERT", "AN_COMERT"]: a[c] = a[c].fillna("").str.strip()
a = a[(a.JUDET_COMERT != "") & (a.NR_COMERT != "") & (a.AN_COMERT != "")]
a["nr"] = pd.to_numeric(a.NR_COMERT, errors="coerce"); a["yr"] = pd.to_numeric(a.AN_COMERT, errors="coerce")
a = a.dropna(subset=["nr", "yr"]); a["nr"] = a.nr.astype(int); a["yr"] = a.yr.astype(int)
a["COD_INMATRICULARE"] = a.JUDET_COMERT + "/" + a.nr.astype(str) + "/" + a.yr.astype(str)
a["rank"] = a.groupby(["JUDET_COMERT", "yr"]).nr.rank(pct=True)
a["reg_date"] = pd.to_datetime(dict(year=a.yr, month=np.clip(np.ceil(a["rank"] * 12), 1, 12).astype(int), day=15), errors="coerce")
name = a.DENUMIRE.fillna("").str.upper().str.replace(".", "", regex=False).str.strip()
a["FORMA_JURIDICA"] = np.where(name.str.endswith(("SRL", "SRL-D", "S R L")), "SRL", np.where(name.str.endswith("SA"), "SA", "OTHER"))
addr = (a.JUDET.fillna("") + "|" + a.LOCALITATE.fillna("") + "|" + a.STRADA.fillna("") + "|" + a.NR.fillna("") + "|" + a.AP.fillna("")).str.upper().str.replace(r"\s+", " ", regex=True)
a["addr_id"] = addr.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
a["radiata"] = a.DATA_RADIERE.fillna("").str.strip() != ""
firms = a[["cui", "COD_INMATRICULARE", "reg_date", "FORMA_JURIDICA", "addr_id", "radiata"]].rename(columns={}).drop_duplicates("cui")
firms["ADR_JUDET"] = a.JUDET
firms.to_parquet(f"{OUT}/onrc_firms.parquet", index=False)
adm = pd.read_parquet(f"{OUT}/onrc_admins.parquet")
link = adm.merge(firms[["cui", "COD_INMATRICULARE", "reg_date"]], on="COD_INMATRICULARE")
link.to_parquet(f"{OUT}/onrc_person_firm.parquet", index=False)
print(f"ANAF PJ with registration code: {len(firms):,} (SRL {(firms.FORMA_JURIDICA=='SRL').sum():,}); admin rows: {len(adm):,}, matched to a CUI: {len(link):,} "
      f"({100*len(link)/len(adm):.1f}%); firms with >=1 admin: {link.cui.nunique():,}")
print("registrations by year (approx):", firms.reg_date.dt.year.value_counts().sort_index().loc[2018:2024].to_dict())
p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui"])
fc = set(firms.cui); s23 = p.loc[p.year == 2023, "cui"]; print(f"2023 statement filers matched to a firm record: {100*s23.isin(fc).mean():.1f}%")

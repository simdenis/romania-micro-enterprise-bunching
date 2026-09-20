"""ONRC linkage. Stage 1: administrators -> person key (name, birth date, birth place) per firm registration code.
Stage 2 (needs OD_FIRME.CSV): map registration code -> CUI, registration date, county, legal form; build
person -> firms table and a firm-level table with: number of firms the administrator(s) run, and dates of those firms.
Personal identifiers are kept only in data/panel/ (never in outputs/); the person key is hashed."""
import os, sys, hashlib, pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = f"{ROOT}/data/raw/onrc/02.09.2026"; OUT = f"{ROOT}/data/panel"
stage = sys.argv[1] if len(sys.argv) > 1 else "1"

if stage == "1":
    r = pd.read_csv(f"{SNAP}/OD_REPREZENTANTI_LEGALI.CSV", sep="^", encoding="utf-8-sig", dtype=str, quoting=3,
                    on_bad_lines="skip", index_col=False, engine="c")
    r.columns = [c.strip() for c in r.columns]
    r = r[r["CALITATE"].str.strip().str.lower().isin(["administrator", "administrator si reprezentant", "reprezentant al persoanei juridice"])]
    for c in ["PERSOANA_IMPUTERNICITA", "DATA_NASTERE", "LOCALITATE_NASTERE"]:
        r[c] = r[c].fillna("").str.strip().str.upper()
    r = r[r["PERSOANA_IMPUTERNICITA"] != ""]
    r["has_birth"] = r["DATA_NASTERE"] != ""
    key = r["PERSOANA_IMPUTERNICITA"] + "|" + r["DATA_NASTERE"] + "|" + r["LOCALITATE_NASTERE"]
    r["person_id"] = key.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    adm = r[["COD_INMATRICULARE", "person_id", "has_birth", "CALITATE"]].drop_duplicates(["COD_INMATRICULARE", "person_id"])
    adm.to_parquet(f"{OUT}/onrc_admins.parquet", index=False)
    n_per = adm.groupby("person_id").size()
    print(f"administrator rows kept: {len(adm):,}; persons: {adm.person_id.nunique():,}; with birth date: {adm.has_birth.mean()*100:.1f}%")
    print("firms per person:", n_per.value_counts().sort_index().head(8).to_dict(), " persons with 2+ firms:", (n_per >= 2).sum())

if stage == "2":
    f = pd.read_csv(f"{SNAP}/OD_FIRME.CSV", sep="^", encoding="utf-8-sig", dtype=str, quoting=3, on_bad_lines="skip",
                    index_col=False, engine="c", usecols=["DENUMIRE", "CUI", "COD_INMATRICULARE", "DATA_INMATRICULARE", "FORMA_JURIDICA", "ADR_JUDET", "ADR_LOCALITATE", "ADR_DEN_STRADA", "ADR_NR_STRADA", "ADR_APARTAMENT"])
    f["cui"] = pd.to_numeric(f["CUI"], errors="coerce"); f = f.dropna(subset=["cui"]); f = f[f.cui > 0]; f["cui"] = f.cui.astype("int64")
    f["reg_date"] = pd.to_datetime(f["DATA_INMATRICULARE"].str.slice(0, 10), format="%d/%m/%Y", errors="coerce")   # some rows carry a time part
    addr = (f["ADR_JUDET"].fillna("") + "|" + f["ADR_LOCALITATE"].fillna("") + "|" + f["ADR_DEN_STRADA"].fillna("") + "|" + f["ADR_NR_STRADA"].fillna("") + "|" + f["ADR_APARTAMENT"].fillna("")).str.upper().str.replace(r"\s+", " ", regex=True)
    f["addr_id"] = addr.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    st = pd.read_csv(f"{SNAP}/OD_STARE_FIRMA.CSV", sep="^", encoding="utf-8-sig", dtype=str, quoting=3, on_bad_lines="skip", index_col=False)
    st.columns = [c.strip() for c in st.columns]
    firms = f[["cui", "COD_INMATRICULARE", "reg_date", "FORMA_JURIDICA", "ADR_JUDET", "addr_id"]].merge(st.rename(columns={"COD": "stare"}), on="COD_INMATRICULARE", how="left")
    firms = firms.drop_duplicates("cui")
    firms.to_parquet(f"{OUT}/onrc_firms.parquet", index=False)
    adm = pd.read_parquet(f"{OUT}/onrc_admins.parquet")
    link = adm.merge(firms[["cui", "COD_INMATRICULARE", "reg_date"]], on="COD_INMATRICULARE").rename(columns={"COD_INMATRICULARE": "code"})
    link.to_parquet(f"{OUT}/onrc_person_firm.parquet", index=False)
    print(f"firms: {len(firms):,} (SRL: {(firms.FORMA_JURIDICA=='SRL').sum():,}); person-firm links: {len(link):,}; firms with >=1 admin: {link.cui.nunique():,}")
    print("registration year counts:", firms.reg_date.dt.year.value_counts().sort_index().loc[2014:2026].to_dict())
    print("legal forms:", firms.FORMA_JURIDICA.value_counts().head(6).to_dict())
    p = pd.read_parquet(f"{ROOT}/data/panel/statements_2014_2025.parquet", columns=["year", "cui"]); lc = set(link.cui)
    print("statement filers with >=1 linked administrator:", p.groupby("year").cui.apply(lambda s: round(100 * s.isin(lc).mean(), 1)).loc[[2019, 2022, 2023, 2024, 2025]].to_dict())

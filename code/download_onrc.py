"""Download ONRC registry snapshots (firms, legal representatives, status, nomenclatures) for chosen dates."""
import json, re, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pk = json.load(open(f"{ROOT}/data/catalog/packages.json"))
WANT = sys.argv[1:] or ["02.09.2026"]
for p in pk:
    for tag in WANT:
        if tag in p["title"] and ("Firme înregistrate" in p["title"] or "Nomenclatoare" in p["title"]):
            d = f"{ROOT}/data/raw/onrc/{tag}"; os.makedirs(d, exist_ok=True)
            for r in p["resources"]:
                dest = f"{d}/{r['name'].upper()}"
                if os.path.exists(dest) and os.path.getsize(dest) > 1000: print("skip", dest, flush=True); continue
                print("get", tag, r["name"], (r.get("size") or 0) // 1_000_000, "MB", flush=True)
                rc = subprocess.run(["curl", "-s", "-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-m", "3600", "-o", dest + ".part", r["url"]]).returncode
                if rc == 0 and os.path.getsize(dest + ".part") > 1000: os.replace(dest + ".part", dest); print("  ok", flush=True)
                else: print("  FAILED", rc, flush=True)

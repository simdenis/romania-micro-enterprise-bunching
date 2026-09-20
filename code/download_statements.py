"""Download Finance Ministry firm-level statements (2014-2025) and the ANAF June-2023 taxpayer file.
For each year and file type (UU = simplified/micro-entity form, BL = long/short form) the most recently
uploaded copy across all catalog datasets is chosen. Writes data/raw/manifest.csv."""
import json, re, csv, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pk = json.load(open(f"{ROOT}/data/catalog/packages.json"))
best = {}
for p in pk:
    if not re.search(r'^Situa.{1,3}i(le)? financiare 20', p['title']): continue
    for r in p['resources']:
        n = r['name'].lower()
        if not n.endswith('.txt'): continue
        m = re.search(r'(uu|bl_bs_sl)_?(an)?(20\d\d)', n)
        if not m: continue
        typ = 'UU' if m.group(1) == 'uu' else 'BL'
        yr = int(m.group(3))
        if not 2014 <= yr <= 2025: continue
        stamp = r.get('last_modified') or r.get('created') or ''
        key = (yr, typ)
        if key not in best or stamp > best[key]['stamp']:
            best[key] = dict(stamp=stamp, url=r['url'], dataset=p['title'], name=r['name'], size=r.get('size'))
jobs = [(f"fin/{yr}_{typ}.txt", v) for (yr, typ), v in sorted(best.items())]
for p in pk:
    if p['title'] == 'Date de identificare plătitori actualizate iunie 2023':
        for r in p['resources']:
            if r['name'].lower().endswith('.csv') and 'platitori' in r['name'].lower():
                jobs.append((f"anaf/2023_{r['name']}", dict(stamp=r.get('last_modified',''), url=r['url'], dataset=p['title'], name=r['name'], size=r.get('size'))))
with open(f"{ROOT}/data/raw/manifest.csv", 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['local_file', 'dataset', 'resource', 'uploaded', 'url'])
    for local, v in jobs: w.writerow([local, v['dataset'], v['name'], v['stamp'][:10], v['url']])
for local, v in jobs:
    dest = f"{ROOT}/data/raw/{local}"
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print('skip', local, flush=True); continue
    print('get ', local, '<-', v['dataset'], '/', v['name'], flush=True)
    rc = subprocess.run(['curl', '-s', '-L', '--retry', '3', '-m', '1800', '-o', dest + '.part', v['url']]).returncode
    if rc == 0 and os.path.getsize(dest + '.part') > 1000:
        os.replace(dest + '.part', dest); print('  ok', os.path.getsize(dest) // 1_000_000, 'MB', flush=True)
    else:
        print('  FAILED rc', rc, flush=True)

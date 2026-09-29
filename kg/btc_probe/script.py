"""Зонд 2: сырые строки BTCUSDT из panel.zip — где ломаются метки."""
import glob, zipfile
zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
rows = zf.read("BTCUSDT_30m.csv").decode().strip().split("\n")
print("всего строк:", len(rows), flush=True)
print("ШАПКА:", rows[0][:100], flush=True)
for r in rows[1:4]:
    print("RAW:", r[:110], flush=True)
print("...", flush=True)
for r in rows[-2:]:
    print("RAW-END:", r[:110], flush=True)
# распределение длин open_time (мс ~13 знаков, мкс ~16)
from collections import Counter
lens = Counter(len(r.split(",")[0]) for r in rows[1:])
print("длины open_time:", dict(lens), flush=True)

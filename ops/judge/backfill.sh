#!/bin/bash
# обратный отсчёт судьи: 1-11 октября
for d in 01 02 03 04 05 06 07 08 09 10 11; do
  echo "=== 2026-10-$d ==="
  python3 ops/daily_judge.py "2026-10-$d" 2>&1 | grep -vE "RuntimeWarning|ot.astype|df\["
done

"""Полный разбор результатов репетиции боя (usdc-battle-replay) для
отчёта владельцу: все метрики на русском, из отчёта кернела и выгрузки
сделок. Запускается на сервере ПОСЛЕ финиша кернела."""
import json
import glob
import math
import statistics
from collections import defaultdict

OUT = "/root/usdc/kg/battle_replay_out"

rep = json.load(open(f"{OUT}/battle_replay_report.json"))
trades = [json.loads(l) for l in open(f"{OUT}/battle_replay_trades.jsonl")]

# ---------- 1. СВОДКА СДЕЛОК ----------
n = len(trades)
wins = [t for t in trades if t["win"]]
losses = [t for t in trades if not t["win"]]
net = [t["net_bp"] for t in trades]
stops = [t for t in trades if t["kind"] == "stop"]
bud = [t for t in trades if t["kind"] == "budilnik"]
deli = [t for t in trades if t["kind"] == "delist"]
best_t = max(trades, key=lambda t: t["net_bp"])
worst_t = min(trades, key=lambda t: t["net_bp"])

print("=== 1. СДЕЛКИ: ОБЩАЯ КАРТИНА ===")
print(f"всего сделок: {n}")
print(f"выигрышных: {len(wins)} ({len(wins)/n*100:.1f}%)")
print(f"проигрышных: {len(losses)} ({len(losses)/n*100:.1f}%)")
print(f"средний итог на сделку: {sum(net)/n:+.1f} пунктов")
med = statistics.median(net)
print(f"медианный итог (половина сделок лучше): {med:+.1f} пунктов")
print(f"средний выигрыш (когда в плюс): {sum(t['net_bp'] for t in wins)/len(wins):+.1f}")
print(f"средний проигрыш (когда в минус): {sum(t['net_bp'] for t in losses)/len(losses):+.1f}")
print(f"лучшая сделка: {best_t['net_bp']:+.0f} пунктов ({best_t['sym']}, {best_t['month']})")
print(f"худшая сделка: {worst_t['net_bp']:+.0f} пунктов ({worst_t['sym']}, {worst_t['month']})")
print(f"стопов (потеря 15%): {len(stops)} ({len(stops)/n*100:.1f}% сделок)")
print(f"вышли по будильнику (5 часов): {len(bud)} ({len(bud)/n*100:.1f}%)")
print(f"закрыты принудительно (конец данных): {len(deli)}")
days_span = (max(t['entry_ms'] for t in trades) - min(t['entry_ms'] for t in trades)) / 86400000
print(f"сделок в день (в среднем): {n/max(1,days_span):.1f}")
fund = sum(t["fund_bp"] for t in trades)
print(f"фандинг всего: {fund:+.0f} пунктов (в среднем {fund/n:+.1f} на сделку)")
fees = sum(t["fee_bp"] for t in trades)
print(f"комиссии всего: {fees:.0f} пунктов ({fees/n:.1f} на сделку)")

# ---------- 2. ПО ГОДАМ ----------
print("\n=== 2. ПО ГОДАМ ===")
years = defaultdict(list)
for t in trades:
    years[t["year"]].append(t)
mrep = {m["month"]: m for m in rep["monthly"]}
bal_by_month = {m["month"]: m["balance"] for m in rep["monthly"]}
print("год | сделок | выигрышей | средняя | сумма пунктов | стопов | баланс на конец (компаунд)")
for y in sorted(years):
    rows = years[y]
    w = sum(1 for t in rows if t["win"])
    avg = sum(t["net_bp"] for t in rows) / len(rows)
    st = sum(1 for t in rows if t["kind"] == "stop")
    # последний месяц года в балансе
    mkeys = sorted(k for k in bal_by_month if k.startswith(str(y)))
    b = bal_by_month[mkeys[-1]] if mkeys else "-"
    print(f"{y} | {len(rows)} | {w} ({w/len(rows)*100:.0f}%) | {avg:+.1f} | "
          f"{sum(t['net_bp'] for t in rows):+.0f} | {st} | {b}")

# ---------- 3. ДЕНЬГИ: ОБЕ КРИВЫЕ ----------
print("\n=== 3. ДЕНЬГИ ===")
mm = rep["money"]
mf = rep["money_fixed"]
print("--- компаунд (номинал растёт с балансом, старт 100) ---")
print(f"итог: {mm['final']:.2f} (вырос в {mm['growth_x']} раза)")
print(f"дней торговали: {mm['days']}")
print(f"максимальная просадка: {mm['maxdd_pct']}% от пика")
print(f"лучший день: +{mm['best_day']:.2f} | худший день: {mm['worst_day']:.2f}")
print(f"красных дней: {mm['red_days']} ({mm['red_pct']}%)")
print(f"самая длинная серия красных дней подряд: {mm['worst_streak']}")
print(f"месяцев: {mm['months']}, плюсовых: {mm['months_pos']} "
      f"({mm['months_pos']/max(1,mm['months'])*100:.0f}%)")
print(f"кап дня срабатывал (блокировок входов): {mm['cap_blocks']}")
print("--- фиксированная сотня (номинал всегда 15, без роста) ---")
print(f"итог: {mf['final']:+.2f}")
print(f"максимальная просадка: {mf['maxdd']:.2f}")
print(f"красных дней: {mf['red_days']}")

# восстановление после худшей просадки (компаунд) по месячным балансам
bals = [m["balance"] for m in rep["monthly"]]
months_names = [m["month"] for m in rep["monthly"]]
peak, peak_i, dd, dd_i = bals[0], 0, 0, 0
worst_dd, worst_dd_after = 0, ""
for i, b in enumerate(bals):
    if b > peak:
        peak, peak_i = b, i
    d = (b - peak) / peak
    if d < worst_dd:
        worst_dd, worst_dd_after = d, months_names[i]
        dd_from = months_names[peak_i]
print(f"худшая просадка по месяцам: {worst_dd*100:.0f}% "
      f"(с пика {dd_from} до дна {worst_dd_after})")
# сколько месяцев ушло на восстановление
trough_i = months_names.index(worst_dd_after)
recovered = next((months_names[i] for i in range(trough_i, len(bals))
                  if bals[i] >= peak), "не восстановился до конца")
print(f"восстановился до пика: {recovered}")

# ---------- 4. КАП И РИСК ----------
print("\n=== 4. ДНИ С БОЛЬШИМИ ДВИЖЕНИЯМИ (компаунд) ===")
day_vals = rep["months_money"]
worst_months = sorted(day_vals.items(), key=lambda kv: kv[1])[:3]
best_months = sorted(day_vals.items(), key=lambda kv: kv[1], reverse=True)[:3]
print("худшие месяцы:", [(m, f"{v:+.0f}") for m, v in worst_months])
print("лучшие месяцы:", [(m, f"{v:+.0f}") for m, v in best_months])

# ---------- 5. ВОРОНКА ----------
print("\n=== 5. ВОРОНКА (кто сколько отрезал) ===")
f = rep["funnel"]
print(f"сырых сигналов (после порога 0.65): {f['cand']:,}")
print(f"мета завернула: {f['meta_veto']:,} ({f['meta_veto']/max(1,f['cand'])*100:.0f}%)")
print(f"тихая треть отрезала: {f['quiet_cut']:,}")
print(f"честный шанс отрезал: {f['honest_skip']:,}")
print(f"стол был полон (сигнал пропал): {f['slot_skip']:,}")
print(f"дневной кап заблокировал входов: {f['cap_block']:,}")
print(f"заявок выставлено: {f['orders']:,}")
print(f"не взялись (цена убежала): {f['unfilled']:,} "
      f"({f['unfilled']/max(1,f['orders'])*100:.1f}% заявок)")
print(f"стало сделок: {n:,}")
print(f"мета включилась с месяца: {rep['meta_on']}")
print(f"карта включилась с месяца: {rep['map_on']}")

# ---------- 6. ПО СТОРОНАМ ----------
print("\n=== 6. ЛОНГИ ПРОТИВ ШОРТОВ ===")
longs = [t for t in trades if t["side"] == "LONG"]
shorts = [t for t in trades if t["side"] == "SHORT"]
for name, rows in (("лонги", longs), ("шорты", shorts)):
    if rows:
        w = sum(1 for t in rows if t["win"])
        print(f"{name}: {len(rows)} сделок | выигрышей {w/len(rows)*100:.1f}% | "
              f"средняя {sum(t['net_bp'] for t in rows)/len(rows):+.1f} пунктов | "
              f"стопов {sum(1 for t in rows if t['kind']=='stop')}")

# ---------- 7. СЕРИИ СДЕЛОК ----------
print("\n=== 7. СЕРИИ (по порядку сделок) ===")
streak_w = streak_l = cur_w = cur_l = 0
for t in trades:
    if t["win"]:
        cur_w += 1; cur_l = 0
    else:
        cur_l += 1; cur_w = 0
    streak_w = max(streak_w, cur_w)
    streak_l = max(streak_l, cur_l)
print(f"самая длинная серия выигрышей подряд: {streak_w}")
print(f"самая длинная серия проигрышей подряд: {streak_l}")

# ---------- 8. РАСПРЕДЕЛЕНИЕ ----------
print("\n=== 8. ЧТО ЗА ЧТО (расклад итогов сделок) ===")
buckets = [(-10**9, -500), (-500, -100), (-100, 0), (0, 100), (100, 300),
           (300, 1000), (1000, 10**9)]
labels = ["хуже −500", "−500..−100", "−100..0", "0..+100", "+100..+300",
          "+300..+1000", "лучше +1000"]
for (lo, hi), lab in zip(buckets, labels):
    c = sum(1 for v in net if lo <= v < hi)
    print(f"{lab:>14}: {c} сделок ({c/n*100:.1f}%)")

# ---------- 9. КОРИДОР / УТЕЧКА ----------
print("\n=== 9. СВЕРКА С ЗАФИКСИРОВАННЫМ ОЖИДАНИЕМ ===")
avg_total = sum(net)/n
print(f"суммарная средняя: {avg_total:+.1f} пунктов")
print("коридор до запуска: +48..+66 пунктов на сделку")
if avg_total > 66:
    print("ВЫШЕ КОРИДОРА — нужно проверять на утечку!")
elif avg_total >= 48:
    print("в коридоре — результат совпал с ожиданием")
else:
    print("НИЖЕ коридора — честный итог: полная цепочка на невиданных "
          "месяцах даёт меньше, чем ожидалось по частичным замерам")

# ---------- 10. ПО МЕСЯЦАМ ----------
print("\n=== 10. ВСЕ МЕСЯЦЫ ПО ПОРЯДКУ ===")
print("месяц | сделок | выигрышей | средняя | баланс на конец")
for m in rep["monthly"]:
    print(f"{m['month']} | {m['trades']} | {m['win']}% | "
          f"{m['avg_bp']:+.1f} | {m['balance']:.2f}")

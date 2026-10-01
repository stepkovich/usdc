"""Кегл-синхронизатор (запускается НА VPS по cron). Полный цикл без
локальной машины:
  1. Заливает свежий архив стакана (parquet) датасетом на Кегл.
  2. Пушит кернел обучения LOB, ждёт, забирает отчёт.
  3. Если отчёт ok (плюс-дней >= половины тестовых) — кладёт модель в
     data/lob/model.txt (движок подхватит по mtime) и шлёт ТГ.
  4. Для широкой ML5H: если кернел панели завершился, а wide ещё не
     пушен — пушит wide-кернел; на вердикте по критериям деплоит модель.
"""
import json, subprocess, sys, time, glob, os
from datetime import datetime, timezone
from pathlib import Path

TOKEN = Path("/root/.kaggle/token").read_text().strip()
ENV = dict(os.environ, KAGGLE_API_TOKEN=TOKEN)
USDC = Path("/root/usdc")
LOB_ARCHIVE = USDC / "data" / "lob" / "archive"
LOB_MODEL = USDC / "data" / "lob" / "model.txt"
FLAGS = USDC / "data" / "kg_flags"
TG_ENV = USDC / ".tg_env"

def tg(text: str) -> None:
    if TG_ENV.exists():
        env = dict(line.strip().split("=", 1) for line in TG_ENV.read_text()
                   .splitlines() if "=" in line)
        import urllib.request
        data = f"chat_id={env['TELEGRAM_CHAT_ID']}&text={text}".encode()
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"https://api.telegram.org/bot{env['TELEGRAM_TOKEN']}/sendMessage",
                data=data), timeout=10)
        except Exception:
            pass

def kaggle(*args, timeout=600) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       capture_output=True, text=True, timeout=timeout, env=ENV)
    return (r.stdout or "") + (r.stderr or "")

def kernel_status(slug: str) -> str:
    out = kaggle("kernels", "status", f"sewerted/{slug}", timeout=60)
    for line in out.splitlines():
        if "has status" in line:
            raw = line.split("has status")[-1].strip().strip('"')
            # CLI 2.x отдаёт "KernelWorkerStatus.COMPLETE" в кавычках
            return raw.split(".")[-1].strip().strip('"').lower()
    return out.strip()[:80]

def disk_guard(threshold: float = 85.0) -> None:
    """Дисковый сторож: если занято > threshold% — чистим ТОЛЬКО то,
    что заведомо не нужно (старые выводы кернелов, кеши, dangling
    образы, WAL-хвост). НИКОГДА не трогаем: код, .git, базы сделок,
    модели, архив паркетов, lob_out (последний отчёт)."""
    st = os.statvfs("/")
    used_pct = 100.0 * (1 - st.f_bfree / st.f_blocks)
    if used_pct < threshold:
        return
    freed = 0
    def _du(path: str) -> int:
        if not os.path.exists(path):
            return 0
        s = subprocess.run(["du", "-sm", path], capture_output=True,
                           text=True)
        try:
            return int(s.stdout.split()[0])
        except Exception:
            return 0
    # 1) старые выводы wide-ядра: wide_out — текущий (не трогаем),
    #    wide_out2 и прочие дубли — безопасно
    for junk in ("/root/kg/wide_out2", "/root/kg/pd_check"):
        sz = _du(junk)
        if sz:
            subprocess.run(["rm", "-rf", junk])
            freed += sz
    # 2) кеш пипа
    sz = _du("/root/.cache/pip")
    if sz:
        subprocess.run(["rm", "-rf", "/root/.cache/pip"])
        freed += sz
    # 3) dangling docker-образы
    try:
        r = subprocess.run(["docker", "image", "prune", "-af"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                if "reclaimed" in line.lower():
                    val = "".join(c for c in line if c.isdigit() or c == ".")
                    if val:
                        freed += int(float(val.split(".")[0]))
    except Exception:
        pass
    # 4) WAL-хвост стакана (безопасно: чекпойнт тот же файл, busy-ожидание)
    db = USDC / "data" / "lob" / "lob.db"
    if db.exists():
        try:
            import sqlite3
            con = sqlite3.connect(str(db), timeout=5)
            con.execute("PRAGMA busy_timeout=5000")
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
        except Exception as e:
            print(f"checkpoint lob.db не удался (не страшно): {e}",
                  flush=True)
    if freed > 20:
        st2 = os.statvfs("/")
        now_pct = 100.0 * (1 - st2.f_bfree / st2.f_blocks)
        tg(f"🧹 Дисковый сторож: было {used_pct:.0f}%, стало {now_pct:.0f}% "
           f"(освобождено ~{freed} МБ)")
    st3 = os.statvfs("/")
    if 100.0 * (1 - st3.f_bfree / st3.f_blocks) > 93:
        tg(f"⚠️ Диск {100.0 * (1 - st3.f_bfree / st3.f_blocks):.0f}% после "
           f"чистки — нужен ручной разбор")

def main() -> None:
    # защита от наложения запусков (часовой cron, кернел ждёт до 2ч)
    lock = Path("/root/kg/kg_sync.lock")
    if lock.exists():
        import time as _t
        if _t.time() - lock.stat().st_mtime < 7200:
            print("предыдущий sync ещё работает — выхожу", flush=True)
            return
        lock.unlink()
    lock.write_text(str(time.time()))
    try:
        _main()
    finally:
        lock.unlink(missing_ok=True)

def _main() -> None:
    FLAGS.mkdir(parents=True, exist_ok=True)
    disk_guard()
    # ---------- 1. LOB: данные -> Кегл ----------
    parquets = sorted(glob.glob(str(LOB_ARCHIVE / "*.parquet")))
    # Датасет на Кегле обновляет ЛОКАЛЬНАЯ машина (у неё полная история —
    # VPS-диск не вмещает 4 дня). Здесь только проверка свежести.
    if parquets:
        fresh_cut = time.time() - 36 * 3600
        stale = [p for p in parquets if os.path.getmtime(p) < fresh_cut]
        if len(stale) > len(parquets) * 0.5:
            tg("⚠️ Стакан: данные на сервере устарели (большинство файлов "
               "старше 36ч) — локальная машина долго выключена?")
    # ---------- 1б. АНАЛИТИК: месячная и сигнальная цепочка переобучения.
    # Правило (SYSTEM_RULES): планово раз в месяц + внепланово, если
    # последние 30 живых сделок в минусе (флаг от движка).
    retrain_flag = USDC / "data" / "signals" / "retrain_needed"
    now_dt = datetime.now(timezone.utc)
    mflag = FLAGS / f"monthly_retrain_{now_dt:%Y-%m}"
    want_monthly = now_dt.day <= 7 and not mflag.exists()
    want_signal = retrain_flag.exists()
    if (want_monthly or want_signal) and (FLAGS / "panel_done").exists():
        # 1) обновляем панель (кернел-загрузчик добавит свежие месяцы)
        kg_dl = Path("/root/kg/panel_downloader")
        if kg_dl.exists():
            out = kaggle("kernels", "push", "-p", str(kg_dl), timeout=120)
            print("push panel downloader:", out.strip()[:100], flush=True)
        # 2) перезапускаем широкую цепочку: сбрасываем флаги завершения
        (FLAGS / "wide_pushed").unlink(missing_ok=True)
        (FLAGS / "wide_done").unlink(missing_ok=True)
        mflag.touch()
        want_signal = False
        reason = "план (месяц)" if now_dt.day <= 7 else "сигнал просадки"
        tg(f"🧠 Переобучение аналитика запущено ({reason}): панель "
           f"обновляется, экзамен решит")
    if retrain_flag.exists():
        retrain_flag.unlink(missing_ok=True)

    # ---------- 2. LOB: кернел обучения ----------
    # Гейт данных: полную историю в датасет Кегла заливает ЛОКАЛЬНАЯ
    # машина (у сервера архив урезан до часов по месту), поэтому сервер
    # проверяет лишь живость потока (2+ свежих файла), а не объём.
    kg_lob = Path("/root/kg/lob_trainer")
    fresh_q = [p for p in parquets
               if time.time() - os.path.getmtime(p) < 3 * 3600]
    if kg_lob.exists() and len(fresh_q) >= 2:
        lob_gate = FLAGS / "lob_last_push"
        can_push = (not lob_gate.exists()
                    or time.time() - lob_gate.stat().st_mtime >= 6 * 3600)
        if can_push:
            out = kaggle("kernels", "push", "-p", str(kg_lob), timeout=120)
            lob_gate.write_text(str(time.time()))
            print("push lob:", out.strip()[:120], flush=True)
        for _ in range(60):            # до 2 часов
            time.sleep(120)
            st = kernel_status("usdc-lob-train")
            print("lob status:", st, flush=True)
            if st in ("complete", "error", "cancelAcknowledged"):
                break
        if st == "complete":
            out = kaggle("kernels", "output", "sewerted/usdc-lob-train",
                         "-p", "/root/kg/lob_out", timeout=600)
            rep = Path("/root/kg/lob_out/lob_report.json")
            if rep.exists():
                r = json.loads(rep.read_text())
                if r.get("ok") and Path("/root/kg/lob_out/lob.txt").exists():
                    import shutil
                    shutil.copy("/root/kg/lob_out/lob.txt", LOB_MODEL)
                    tg("🧠✅ Стакан: модель переобучена на Кегле и УЖЕ "
                       f"на сервере ({r.get('plus_days')}/{r.get('n_test')} "
                       "плюс-дней)")
                else:
                    reason = r.get("reason") or (
                        "плюс-дней %s из %s" % (r.get("plus_days"), r.get("n_test")))
                    tg("🧠⚠️ Стакан: критерии не взяты (" + reason +
                       ") — работаем на старой модели")
    # ---------- 3. Wide ML5H: панель готова? -> пуш wide ----------
    if not (FLAGS / "panel_done").exists():
        st = kernel_status("usdc-panel-downloader")
        print("panel status:", st, flush=True)
        if st == "complete":
            (FLAGS / "panel_done").touch()
            tg("📦 Панель 527 монет скачана на Кегл — стартует широкое "
               "переобучение (walk-forward + нуль-тест)")
    if (FLAGS / "panel_done").exists() and not (FLAGS / "wide_pushed").exists():
        # пушим широкую ТОЛЬКО когда загрузчик панели реально завершился
        # (иначе повторяем старую ошибку: обучение на пустой панели)
        dl_status = kernel_status("usdc-panel-downloader")
        if dl_status != "complete":
            print(f"панель ещё качается ({dl_status}) — wide отложен",
                  flush=True)
        else:
            kg_wide = USDC / "kg" / "ml5h_wide"
            if (kg_wide / "script.py").exists():
                kaggle("kernels", "push", "-p", str(kg_wide), timeout=120)
                (FLAGS / "wide_pushed").touch()
                (FLAGS / "wide_done").unlink(missing_ok=True)
                tg("🌙 Широкая модель: кернел запущен на Кегле")
    if (FLAGS / "wide_pushed").exists() and not (FLAGS / "wide_done").exists():
        st = kernel_status("usdc-ml5h-wide")
        print("wide status:", st, flush=True)
        if st in ("complete", "error"):
            out = kaggle("kernels", "output", "sewerted/usdc-ml5h-wide",
                         "-p", "/root/kg/wide_out", timeout=1200)
            # вердикт: ИЗ META (пишется всегда при успехе обучения);
            # report.json — дубликат, в старых версиях без поля
            verdict = None
            meta_p = Path("/root/kg/wide_out/ml5h_meta.json")
            if meta_p.exists():
                verdict = json.load(open(meta_p)).get("verdict")
            else:
                rep_p = Path("/root/kg/wide_out/report.json")
                if rep_p.exists():
                    verdict = json.load(open(rep_p)).get("verdict")
            model_p = Path("/root/kg/wide_out/ml5h.txt")
            if verdict and model_p.exists():
                import shutil
                shutil.copy(model_p, USDC / "data" / "models" / "ml5h.txt")
                shutil.copy(meta_p, USDC / "data" / "models" / "ml5h_meta.json")
                (FLAGS / "wide_done").touch()
                tg("🧠✅ ШИРОКАЯ МОДЕЛЬ ПРИНЯТА по критериям и на сервере")
            else:
                (FLAGS / "wide_done").touch()
                tg("🧠⚠️ Широкая модель НЕ прошла критерии — осталась "
                   "текущая; детали в отчёте Кегла")

if __name__ == "__main__":
    main()

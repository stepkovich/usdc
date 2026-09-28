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
            return line.split("has status")[-1].strip()
    return out.strip()[:80]

def main() -> None:
    FLAGS.mkdir(parents=True, exist_ok=True)
    # ---------- 1. LOB: данные -> Кегл ----------
    parquets = sorted(glob.glob(str(LOB_ARCHIVE / "*.parquet")))
    if len(parquets) >= 12:            # хотя бы полдня данных
        meta = {"title": "USDC LOB features",
                "id": "sewerted/usdc-lob-features",
                "licenses": [{"name": "CC0-1.0"}]}
        (LOB_ARCHIVE / "dataset-metadata.json").write_text(json.dumps(meta))
        out = kaggle("datasets", "create", "-p", str(LOB_ARCHIVE),
                     "--dir-mode", "zip", timeout=1800)
        if "successfully" not in out.lower():
            out2 = kaggle("datasets", "version", "-p", str(LOB_ARCHIVE),
                          "-m", "daily sync", "--dir-mode", "zip",
                          timeout=1800)
            out = out2 if "successfully" in out2.lower() else out
        print("dataset:", out.strip()[:120], flush=True)
    # ---------- 2. LOB: кернел обучения (только если данных >= полдня) ----------
    kg_lob = Path("/root/kg/lob_trainer")
    if kg_lob.exists() and len(parquets) >= 12:
        out = kaggle("kernels", "push", "-p", str(kg_lob), timeout=120)
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
                    tg(f"🧠⚠️ Стакан: кернел прошёл, но критерии не взяты "
                       f"({r.get('reason') or r.get('plus_days')} "
                       f"из {r.get('n_test')}) — работаем на старой модели")
    # ---------- 3. Wide ML5H: панель готова? -> пуш wide ----------
    if not (FLAGS / "panel_done").exists():
        st = kernel_status("usdc-panel-downloader")
        print("panel status:", st, flush=True)
        if st == "complete":
            (FLAGS / "panel_done").touch()
            tg("📦 Панель 527 монет скачана на Кегл — стартует широкое "
               "переобучение (walk-forward + нуль-тест)")
    if (FLAGS / "panel_done").exists() and not (FLAGS / "wide_pushed").exists():
        kg_wide = USDC / "kg" / "ml5h_wide"
        if (kg_wide / "script.py").exists():
            kaggle("kernels", "push", "-p", str(kg_wide), timeout=120)
            (FLAGS / "wide_pushed").touch()
            tg("🌙 Широкая модель: кернел запущен на Кегле")
    if (FLAGS / "wide_pushed").exists() and not (FLAGS / "wide_done").exists():
        st = kernel_status("usdc-ml5h-wide")
        print("wide status:", st, flush=True)
        if st in ("complete", "error"):
            out = kaggle("kernels", "output", "sewerted/usdc-ml5h-wide",
                         "-p", "/root/kg/wide_out", timeout=1200)
            rep = Path("/root/kg/wide_out/report.json")
            if rep.exists():
                r = json.loads(rep.read_text())
                if r.get("verdict") and Path("/root/kg/wide_out/ml5h.txt").exists():
                    import shutil
                    shutil.copy("/root/kg/wide_out/ml5h.txt",
                                USDC / "data" / "models" / "ml5h.txt")
                    shutil.copy("/root/kg/wide_out/ml5h_meta.json",
                                USDC / "data" / "models" / "ml5h_meta.json")
                    (FLAGS / "wide_done").touch()
                    tg("🧠✅ ШИРОКАЯ МОДЕЛЬ ПРИНЯТА по критериям и на сервере")
                else:
                    (FLAGS / "wide_done").touch()
                    tg("🧠⚠️ Широкая модель НЕ прошла критерии — осталась "
                       "текущая; детали в отчёте Кегла")

if __name__ == "__main__":
    main()

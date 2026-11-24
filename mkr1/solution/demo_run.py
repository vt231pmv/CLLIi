"""Один запуск = усі AI-докази + автозбірка звіту (його ж зручно записувати на відео демо).

  1) python -m labcheck doctor            — ключ і модель працюють?
  2) бенчмарк підкинутих дефектів (без AI) — evidence/bench_static.json
  3) AI-бенчмарк: розбір 3 завдань + суддівство 50 відповідей — evidence/ai_run/bench_ai.json
  4) реальне використання з AI: перевірка самої папки mkr1 (12 відповідей звіту проти питань МКР-1)
     і, якщо задано --lab6 ТЕКА, вашої лаби 6
  5) finalize_report.py — підставляє виміряні числа в report.md

Параметри: --fresh (ігнорувати кеш відповідей AI), --lab6 ТЕКА, --no-ai (лише кроки 2 і 5).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MKR = HERE.parent
EV = MKR / "evidence"
sys.path.insert(0, str(HERE))

from labcheck import bench  # noqa: E402
from labcheck.cli import _utf8_console, cmd_doctor, main as labcheck_main  # noqa: E402
from labcheck.llm import LLMClient, load_dotenv  # noqa: E402


class Tee(io.TextIOBase):
    def __init__(self, *outs):
        self.outs = outs

    def write(self, s):
        for o in self.outs:
            o.write(s)
        return len(s)

    def flush(self):
        for o in self.outs:
            try:
                o.flush()
            except ValueError:  # файл журналу вже закрито (завершення програми) — не страшно
                pass


def step(n, total, title):
    print(f"\n=== Крок {n}/{total}: {title} ===", flush=True)


def run_check(args_list, out_txt: Path) -> int:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = labcheck_main(args_list)
    text = buf.getvalue()
    out_txt.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return code


def main() -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="не брати відповіді з кешу")
    ap.add_argument("--lab6", help="тека lab6 з вашими ноутбуками (необов'язково)")
    ap.add_argument("--no-ai", action="store_true")
    a = ap.parse_args()
    (EV / "ai_run").mkdir(parents=True, exist_ok=True)
    log = (EV / "ai_run" / "demo_run.log").open("w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log)
    try:
        t0 = time.perf_counter()
        total = 5
        print(f"labcheck demo_run | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        load_dotenv()
        llm = None if a.no_ai else LLMClient.from_env(cache_dir=None if a.fresh else HERE / ".labcheck_cache")
        if not a.no_ai:
            step(1, total, "доступ до AI")
            if cmd_doctor(None) != 0 or llm is None:
                print("\nAI недоступний. Виправте .env (див. .env.example) і повторіть, або запустіть з --no-ai.")
                return 2
        step(2, total, "підкинуті дефекти (без AI)")
        r = bench.run_static_bench(5)
        bench._print_static(r)
        (EV / "bench_static.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        if llm is not None:
            step(3, total, "AI-бенчмарк: розбір завдань і суддівство відповідей")
            res = {"model": llm.model, "date": time.strftime("%Y-%m-%d")}
            res["spec"] = bench.run_ai_spec_bench(llm)
            for n, v in res["spec"].items():
                shown = {k: v[k] for k in v if k not in ("verification", "tokens")}
                print(f"[розбір {n}]", json.dumps(shown, ensure_ascii=False))
            res["judge"] = bench.run_ai_judge_bench(llm)
            print("[суддівство]", json.dumps({k: v for k, v in res["judge"].items() if k not in ("verdicts", "per_kind")}, ensure_ascii=False))
            res["llm_total"] = llm.stats.as_dict()
            print("[разом]", json.dumps(res["llm_total"], ensure_ascii=False))
            bad = [n for n, v in res["spec"].items() if not v.get("ok")] + (["суддівство"] if res["judge"]["failed_batches"] else [])
            if bad:
                print(f"\nУвага: не вдалося для: {', '.join(bad)} (сервіс моделі недоступний). Просто запустіть `python demo_run.py` ще раз "
                      "БЕЗ --fresh: успішні відповіді візьмуться з кешу, повторяться лише невдалі.")
            (EV / "ai_run" / "bench_ai.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            step(4, total, "реальне використання з AI-суддею")
            import finalize_report
            finalize_report.render()  # звіт має існувати й містити виміряні числа до самоперевірки
            base = ["--spec", str(HERE / "specs" / "mkr1.json"), "--history", str(EV / "history.jsonl")]
            if not (MKR / "demo.mp4").exists():
                print("(demo.mp4 ще немає — помилка E_MISSING очікувана, поки ви не збережете запис)")
            run_check(["check", str(MKR), *base, "--label", "mkr1_self_check_ai", "--md", str(EV / "ai_run" / "mkr1_self_check.md")] + (["--no-cache"] if a.fresh else []),
                      EV / "ai_run" / "mkr1_self_check.txt")
            if a.lab6:
                run_check(["check", a.lab6, "--spec", str(HERE / "specs" / "lab6.json"), "--history", str(EV / "history.jsonl"),
                           "--label", "lab6_user_ai", "--md", str(EV / "ai_run" / "lab6_check.md")] + (["--no-cache"] if a.fresh else []),
                          EV / "ai_run" / "lab6_check.txt")
        step(5, total, "збираю report.md із виміряних чисел")
        import finalize_report
        finalize_report.render()
        finalize_report.check()
        try:
            import make_chart
            make_chart.main()
        except Exception as e:  # matplotlib необов'язковий: без нього лишається попередній графік
            print(f"графік не оновлено ({type(e).__name__}); потрібен matplotlib")
        print(f"\nГотово за {time.perf_counter() - t0:.0f} с. Далі: запишіть demo.mp4 і зробіть git push (див. solution/README.md).")
        return 0
    finally:
        sys.stdout = real_stdout
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())

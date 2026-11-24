"""Збирає report.md із шаблону report.template.md: підставляє числа з логів і вимірювань (нічого не набирається вручну).

  python finalize_report.py            # бере evidence/ai_run/bench_ai.json, якщо він є; інакше AI-частини позначає «не вимірювалось»
  python finalize_report.py --check    # лише перевірити, що в report.md не лишилось {{плейсхолдерів}}
Числа «до» (оцінки без таймера) беруться з estimates.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MKR = HERE.parent
EV = MKR / "evidence"
NA = "не вимірювалось (AI-прогін не виконано: запустіть `python demo_run.py`)"


def c(x, nd=1):
    s = f"{x:.{nd}f}".replace(".", ",")
    return s.rstrip("0").rstrip(",") if "," in s else s


def pct(x):
    return f"{x * 100:.0f}%" if abs(x * 100 - round(x * 100)) < 0.05 else f"{x * 100:.1f}%".replace(".", ",")


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def history() -> dict[str, dict]:
    out = {}
    p = EV / "history.jsonl"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("label"):
                out[d["label"]] = d  # лишається останній запуск із такою міткою
    return out


def placeholders_in_log(label: str) -> int:
    p = EV / "real_runs" / f"{label}.txt"
    if not p.is_file():
        return 0
    return sum(int(m) for m in re.findall(r"плейсхолдерів у markdown-клітинках: (\d+)", p.read_text(encoding="utf-8")))


def build_tokens() -> dict[str, str]:
    est = load(HERE / "estimates.json") or {}
    bs = load(EV / "bench_static.json")
    tm = load(EV / "timing.json") or {}
    ai = load(EV / "ai_run" / "bench_ai.json")
    hist = history()
    T: dict[str, str] = {}

    # --- підкинуті дефекти
    if bs:
        T["BENCH_RECALL"], T["BENCH_FOUND"], T["BENCH_MUTANTS"] = pct(bs["recall"]), str(bs["found"]), str(bs["mutants"])
        T["BENCH_KINDS"], T["BENCH_VARIANTS"] = str(bs["kinds"]), str(bs["variants_per_kind"])
        T["BENCH_FP"], T["BENCH_PRECISION"] = str(bs["findings_on_clean"]), pct(bs["precision_of_findings"])
        rows = ["| Підкинутий дефект | Очікуваний код | Знайдено |", "|---|---|---|"]
        rows += [f"| {r['defect']} | `{r['expected']}` | {r['found']} з {r['variants']} |" for r in bs["rows"]]
        T["BENCH_TABLE"] = "\n".join(rows)
    else:
        for k in ("BENCH_RECALL", "BENCH_FOUND", "BENCH_MUTANTS", "BENCH_KINDS", "BENCH_VARIANTS", "BENCH_FP", "BENCH_PRECISION", "BENCH_TABLE"):
            T[k] = "не вимірювалось (немає evidence/bench_static.json)"

    # --- реальні прогони
    rows = ["| Запуск | Помилок | Попереджень | Плейсхолдерів у клітинках | Питань знайдено | Час, с |", "|---|---|---|---|---|---|"]
    names = {"lab6_before": "Лаба 6, ноутбуки зі Colab (до доопрацювання)", "lab6_after": "Лаба 6, фінальна версія",
             "lab5_before": "Лаба 5, ноутбуки зі Colab (до доопрацювання)", "lab5_after": "Лаба 5, фінальна версія"}
    for k, n in names.items():
        h = hist.get(k)
        if h:
            rows.append(f"| {n} | {h['errors']} | {h['warnings']} | {placeholders_in_log(k)} | {h['questions_found']} | {c(h['duration_s'], 2)} |")
    T["REAL_RUNS_TABLE"] = "\n".join(rows) if len(rows) > 2 else "не вимірювалось (немає evidence/history.jsonl)"
    t_rows = ["| Запуск | Вхід, МБ | Повторів | Середній час, с | Медіана, с | Максимум, с |", "|---|---|---|---|---|---|"]
    for k, n in names.items():
        t = tm.get(k)
        if t:
            t_rows.append(f"| {n} | {c(t['input_mb'], 2)} | {t['repeats']} | {c(t['mean_s'], 3)} | {c(t['median_s'], 3)} | {c(t['max_s'], 3)} |")
    T["TIMING_TABLE"] = "\n".join(t_rows) if len(t_rows) > 2 else "не вимірювалось"
    static_s = sum(t["mean_s"] for t in tm.values()) / len(tm) if tm else 0.0
    T["TOOL_SECONDS"] = c(static_s, 2)
    T["PLACEHOLDERS_LAB6"], T["PLACEHOLDERS_LAB5"] = str(placeholders_in_log("lab6_before")), str(placeholders_in_log("lab5_before"))

    # --- AI
    judge_b = ai.get("judge") if ai else None
    spec_b = ai.get("spec") if ai else None
    ai_time_lab = ai_cost_lab = ai_tok_lab = ai_wall_lab = None
    if judge_b:
        batches = max(1, judge_b["tokens"]["calls"] + judge_b["tokens"]["cache_hits"])
        ai_tok_lab = (judge_b["tokens"]["prompt_tokens"] + judge_b["tokens"]["completion_tokens"]) / batches * 6
        ai_cost_lab = judge_b["tokens"]["cost_usd"] / batches * 6
        # чистий час запитів; старі bench_ai.json без цього поля мають лише час на стіні (з паузами між повторами)
        ai_time_lab = judge_b.get("request_latency_s", judge_b["latency_s"]) / batches * 6
        ai_wall_lab = judge_b["latency_s"] / batches * 6
    T["AI_MODEL"] = f"`{ai['model']}`" if ai else "не визначено (AI-прогін ще не виконано: `python demo_run.py`)"
    if ai and (spec_b or judge_b):
        L = []
        if spec_b:
            L += ["**Розбір тексту завдання (AI) проти еталонних специфікацій:**", "",
                  "| Завдання | Питань: еталон / видобуто | Файли: precision / recall | Текст питань точний | Файл питання вказано вірно | Секції (recall) | Відкинуто як вигадане | Токенів |",
                  "|---|---|---|---|---|---|---|---|"]
            for n, v in spec_b.items():
                if not v.get("ok"):
                    L.append(f"| {n} | збій: {v.get('error', '?')} | | | | | | |")
                    continue
                dropped = sum(v["verification"][k] for k in ("required_dropped", "questions_dropped", "sections_dropped", "forbidden_dropped"))
                L.append(f"| {n} | {v['questions_gold']} / {v['questions_extracted']} | {pct(v['required_precision'])} / {pct(v['required_recall'])} | "
                         f"{pct(v['question_text_ok'])} | {pct(v['question_file_ok'])} | {pct(v['sections_recall'])} | {dropped} | {v['tokens']['prompt_tokens'] + v['tokens']['completion_tokens']} |")
            L.append("")
        if judge_b:
            j = judge_b
            L += [f"**Суддівство відповідей ({j['items']} випадків: 30 реальних відповідей лаби 6, 10 підмін на відповідь іншого питання, 10 відписок без змісту):**", "",
                  "| Показник | Значення |", "|---|---|",
                  f"| Реальних відповідей визнано «є відповідь» | {pct(j['originals_answered_rate'])} |",
                  f"| Хибних тривог на реальних відповідях (не за темою / немає) | {pct(j['originals_false_alarm_rate'])} |",
                  f"| Підміни знайдено | {pct(j['swapped_caught_rate'])} |",
                  f"| Відписки знайдено | {pct(j['filler_caught_rate'])} |",
                  f"| Пропущено дефектів (із 20) | {j['defects_missed']} |",
                  f"| Вердиктів без підтвердженої цитати (код не прийняв, «перевірити вручну») | {j['quote_rejected']} |",
                  f"| Запитів до моделі / з кешу | {j['tokens']['calls']} / {j['tokens']['cache_hits']} |",
                  f"| Токени (вхід + вихід) | {j['tokens']['prompt_tokens']} + {j['tokens']['completion_tokens']} |",
                  f"| Вартість за тарифом із `.env` | ${c(j['tokens']['cost_usd'], 4)} |",
                  f"| Час запитів до моделі (без пауз між повторами) | {c(j.get('request_latency_s', j['latency_s']), 1)} с |",
                  f"| Час на стіні разом із паузами між повторами при перевантаженні сервісу | {c(j['latency_s'], 1)} с |"]
        T["AI_BLOCK"] = "\n".join(L)
        if judge_b:
            j = judge_b
            T["AI_SUMMARY"] = (f"На бенчмарку з {j['items']} випадків AI-суддя пропустив {j['defects_missed']} із 20 підкинутих дефектів (підмін і відписок) "
                               f"і хибно відхилив {c(j['originals_false_alarm_rate'] * 30, 0)} із 30 реальних відповідей; "
                               f"у {j['quote_rejected']} випадках код не прийняв вердикт, бо цитату моделі не знайшов у тексті")
            T["AI_COST_PER_LAB"] = f"≈ ${c(ai_cost_lab, 4)} (≈ {c(ai_tok_lab, 0)} токенів; тариф із `.env`, $0 для безкоштовного рівня)"
            T["AI_TIME_PER_LAB"] = (f"{c(ai_time_lab, 1)} с чистого часу запитів (6 запитів по 5 питань; бенчмарк)"
                                    + (f"; коли сервіс перевантажений і клієнт чекає між повторами, на стіні це було ≈ {c(ai_wall_lab, 0)} с" if ai_wall_lab and ai_wall_lab > 1.5 * ai_time_lab else ""))
        else:
            T["AI_SUMMARY"] = T["AI_COST_PER_LAB"] = T["AI_TIME_PER_LAB"] = "не вимірювалось (запущено без judge)"
        probs = []
        if judge_b and judge_b["failed_batches"]:
            probs.append(f"{judge_b['failed_batches']} пакет(ів) суддівства завершились збоєм (сервіс/формат) — відповіді позначено «перевірити вручну»")
        if judge_b and judge_b.get("request_latency_s") is not None and judge_b["latency_s"] > 1.5 * judge_b["request_latency_s"] + 5:
            probs.append(f"сервіс моделі був перевантажений: суддівство тривало {c(judge_b['latency_s'], 0)} с на стіні проти {c(judge_b['request_latency_s'], 0)} с чистих запитів (решта — паузи між повторами після HTTP 503/429)")
        if judge_b and judge_b["quote_rejected"]:
            probs.append(f"{judge_b['quote_rejected']} вердикт(ів) відхилено захистом від вигадок: цитати немає в тексті")
        if spec_b:
            for n, v in spec_b.items():
                if v.get("ok"):
                    dr = sum(v["verification"][k] for k in ("required_dropped", "questions_dropped", "sections_dropped", "forbidden_dropped"))
                    if v["verification"].get("paths_normalized"):
                        probs.append(f"розбір «{n}»: код привів до повної форми {v['verification']['paths_normalized']} шлях(ів) від моделі (без першої папки або з «/*»)")
                    if dr or v["verification"]["questions_healed"]:
                        probs.append(f"розбір «{n}»: відкинуто вигаданих елементів {dr}, виправлено перефразованих питань {v['verification']['questions_healed']}")
                    if v.get("sections_missing"):
                        probs.append(f"розбір «{n}»: не знайдено секцій {', '.join('«' + x + '»' for x in v['sections_missing'])}")
                    if v["question_file_ok"] < 1:
                        probs.append(f"розбір «{n}»: файл для питань вказано вірно лише у {pct(v['question_file_ok'])}")
                else:
                    probs.append(f"розбір «{n}» не вдався: {v.get('error')}")
        T["AI_PROBLEMS"] = "; ".join(probs) if probs else "у прогоні AI-кроків збоїв не було, захист від вигадок нічого не відхиляв"
    else:
        for k in ("AI_BLOCK", "AI_SUMMARY", "AI_COST_PER_LAB", "AI_TIME_PER_LAB", "AI_PROBLEMS"):
            T[k] = NA

    # --- до / після
    man, read_min = est.get("manual_check_minutes", 20), est.get("reading_output_minutes_after", 2)
    after = read_min + static_s / 60 + ((ai_time_lab or 0) / 60)
    saved = man - after
    T.update({"EST_MANUAL_MIN": c(man, 0), "EST_MANUAL_OPS": str(est.get("manual_operations", 7)), "EST_READ_MIN": c(read_min, 0),
              "EST_OPS_AFTER": str(est.get("operations_after", 1)), "CHECKS_PER_SEMESTER": str(est.get("checks_per_semester", 12)),
              "TIME_AFTER_MIN": c(after, 1), "SAVED_MIN": c(saved, 1), "SAVED_PCT": f"{saved / man * 100:.0f}%",
              "SAVED_SEMESTER_MIN": c(saved * est.get("checks_per_semester", 12), 0)})
    def ok(cond):
        return "так" if cond else "ні"
    rows = ["| KPI | Ціль | Результат | Досягнуто? | Тип значення |", "|---|---|---|---|---|",
            f"| Час на перевірку однієї лаби | ≤ 5 хв | {T['TIME_AFTER_MIN']} хв | {ok(after <= 5)} | оцінка (читання виводу) + виміряно (інструмент{', AI' if ai_time_lab else ''}) |"]
    if bs:
        rows.append(f"| Знаходить підкинуті дефекти (recall) | ≥ 95% | {pct(bs['recall'])} ({bs['found']} із {bs['mutants']}) | {ok(bs['recall'] >= 0.95)} | виміряно |")
        rows.append(f"| Зайві спрацювання на чистому репозиторії | 0 | {bs['findings_on_clean']} | {ok(bs['findings_on_clean'] == 0)} | виміряно |")
    if judge_b:
        caught = (judge_b["swapped_caught_rate"] * 10 + judge_b["filler_caught_rate"] * 10) / 20
        rows.append(f"| AI-суддя знаходить підкинуті порожні/чужі відповіді | ≥ 90% | {pct(caught)} | {ok(caught >= 0.9)} | виміряно (AI-прогін) |")
        rows.append(f"| Вартість перевірки однієї лаби | ≤ $0,01 | ≈ ${c(ai_cost_lab, 4)} | {ok(ai_cost_lab <= 0.01)} | виміряно (токени) × тариф із `.env` |")
    else:
        rows.append(f"| AI-суддя знаходить підкинуті порожні/чужі відповіді | ≥ 90% | {NA} | — | — |")
        rows.append(f"| Вартість перевірки однієї лаби | ≤ $0,01 | {NA} | — | — |")
    T["TARGETS_TABLE"] = "\n".join(rows)
    return T


def render(no_write=False) -> int:
    tpl = (HERE / "report.template.md").read_text(encoding="utf-8")
    T = build_tokens()
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)) - set(T))
    if missing:
        print("У шаблоні невідомі плейсхолдери:", missing, file=sys.stderr)
        return 1
    out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: T[m.group(1)], tpl)
    (MKR / "report.md").write_text(out, encoding="utf-8")
    print(f"report.md зібрано ({len(out)} символів). AI-частина: {'є' if 'AI_BLOCK' in T and T['AI_BLOCK'] != NA else 'не виміряна'}")
    return 0


def check() -> int:
    p = MKR / "report.md"
    if not p.is_file():
        print("Немає report.md", file=sys.stderr)
        return 1
    bad = re.findall(r"\{\{[A-Z0-9_]+\}\}|TODO\(student\)", p.read_text(encoding="utf-8"))
    na = p.read_text(encoding="utf-8").count("не вимірювалось")
    print(f"плейсхолдерів: {len(bad)}; місць «не вимірювалось»: {na}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    raise SystemExit(check() if a.check else render())

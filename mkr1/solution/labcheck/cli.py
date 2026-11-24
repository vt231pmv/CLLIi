"""Командний рядок: check | extract-spec | doctor | bench."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .ai_steps import extract_spec, judge_questions
from .llm import LLMAuthError, LLMClient, LLMUnavailable, load_dotenv
from .notebook import read_text_any
from .report import CheckReport, history_line, render_md, render_text, to_json
from .spec import SpecError, load_spec
from .static_checks import run_static

SOLUTION = Path(__file__).resolve().parents[1]


def _utf8_console() -> None:
    """Windows-консолі часто мають cp1251/cp866: без цього кирилиця або спецсимволи можуть кинути UnicodeEncodeError."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def make_llm(args) -> LLMClient | None:
    if getattr(args, "no_ai", False):
        return None
    load_dotenv()
    cache = None if getattr(args, "no_cache", False) else Path(getattr(args, "cache_dir", None) or SOLUTION / ".labcheck_cache")
    return LLMClient.from_env(cache_dir=cache)


def cmd_check(args) -> int:
    t0 = time.perf_counter()
    llm = make_llm(args)
    notes: list[str] = []
    spec, source, spec_stats = None, "", None
    if args.assignment:
        if llm is None:
            notes.append("AI-розбір завдання пропущено: не задано LLM_API_KEY/LLM_MODEL (або --no-ai)")
        else:
            try:
                text = read_text_any(Path(args.assignment))
                spec, spec_stats = extract_spec(llm, text)
                source = f"AI-розбір {args.assignment}"
                notes.append(f"AI-розбір: прийнято вимог до файлів {spec_stats['required_kept']}, питань {spec_stats['questions_kept']}; "
                             f"відкинуто вигаданого: {spec_stats['required_dropped'] + spec_stats['questions_dropped'] + spec_stats['sections_dropped']}")
            except (LLMUnavailable, ValueError, OSError) as e:
                notes.append(f"AI-розбір не вдався ({type(e).__name__}: {e}); використано запасну специфікацію з файлу")
    if spec is None:
        if not args.spec:
            print("Потрібна --spec (JSON) або працюючий AI-розбір з --assignment.", file=sys.stderr)
            return 2
        try:
            spec = load_spec(args.spec)
        except SpecError as e:
            print(f"Помилка специфікації: {e}", file=sys.stderr)
            return 2
        source = args.spec
    root = Path(args.root)
    if not root.is_dir():
        print(f"Немає теки: {root}", file=sys.stderr)
        return 2
    res = run_static(root, spec)
    rep = CheckReport(spec, source, res, 0.0, spec_stats, notes=notes)
    if llm is not None and res.questions and not args.no_judge:
        try:
            rep.judge = judge_questions(llm, res.questions, batch=args.batch)
        except Exception as e:  # будь-який збій AI не має ламати детерміновану перевірку
            notes.append(f"AI-суддівство пропущено ({type(e).__name__}: {e})")
    elif llm is None and res.questions and not args.assignment:
        notes.append("AI-суддівство пропущено: не задано LLM_API_KEY/LLM_MODEL (або --no-ai); решта перевірок виконано")
    rep.llm_stats = llm.stats.as_dict() if llm is not None else None
    rep.duration_s = time.perf_counter() - t0
    out = render_text(rep)
    sys.stdout.write(out)
    if args.md:
        Path(args.md).write_text(render_md(rep), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(rep), ensure_ascii=False, indent=1), encoding="utf-8")
    if args.history:
        p = Path(args.history)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(history_line(rep, args.label or "") + "\n")
    return 1 if rep.errors else 0


def cmd_extract(args) -> int:
    llm = make_llm(args)
    if llm is None:
        print("Потрібні LLM_API_KEY і LLM_MODEL (див. .env.example).", file=sys.stderr)
        return 2
    try:
        spec, stats = extract_spec(llm, read_text_any(Path(args.assignment)))
    except (LLMUnavailable, ValueError) as e:
        print(f"Не вдалося: {e}", file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Збережено {args.out}: файлів {len(spec['required'])}, питань {len(spec['questions'])}, секцій {len(spec['sections'])}")
    print("Перевірка проти тексту:", json.dumps(stats, ensure_ascii=False))
    print("Токени:", json.dumps(llm.stats.as_dict(), ensure_ascii=False))
    return 0


def cmd_doctor(args) -> int:
    import os
    load_dotenv()
    llm = LLMClient.from_env()
    if llm is None:
        key = os.environ.get("LLM_API_KEY", "")
        if key:  # ключ є, а моделі немає: покажемо, які моделі доступні, щоб вибрати
            from .llm import DEFAULT_BASE_URL
            tmp = LLMClient(os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL), key, "")
            try:
                names = tmp.list_models()
                print("Ключ прийнято, але не задано LLM_MODEL. Доступні моделі (перші 40):")
                print("\n".join("  " + n for n in names[:40]))
            except LLMUnavailable as e:
                print(f"Ключ не підійшов або сервіс недоступний: {e}")
            return 1
        print("Не задано LLM_API_KEY і LLM_MODEL. Скопіюйте .env.example у .env і заповніть.")
        print("Без AI програма все одно працює: python -m labcheck check <тека> --spec specs/lab6.json --no-ai")
        return 1
    print(f"Сервіс: {llm.base_url}\nМодель: {llm.model}")
    try:
        models = llm.list_models()
        print(f"Ключ прийнято, моделей у списку: {len(models)}" + ("" if llm.model in models or not models else f"  УВАГА: «{llm.model}» немає в списку"))
        r = llm.chat("Відповідай одним словом.", "Скажи: ок", 64)
        print(f"Тестовий запит пройшов: «{r.text.strip()[:30]}», {r.prompt_tokens}+{r.completion_tokens} токенів")
        return 0
    except LLMAuthError as e:
        print(f"Ключ не підійшов: {e}")
    except LLMUnavailable as e:
        print(f"Сервіс недоступний: {e}")
    return 1


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(prog="labcheck", description="Перевірка лабораторної перед здачею")
    ap.add_argument("--version", action="version", version=f"labcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="перевірити теку з роботою")
    c.add_argument("root", help="корінь репозиторію або тека lab6/mkr1")
    c.add_argument("--spec", help="JSON-специфікація (specs/lab6.json)")
    c.add_argument("--assignment", help="текст завдання; AI розбере його в специфікацію (запасний варіант — --spec)")
    c.add_argument("--no-ai", action="store_true", help="лише детерміновані перевірки")
    c.add_argument("--no-judge", action="store_true", help="не оцінювати відповіді на питання через AI")
    c.add_argument("--batch", type=int, default=5, help="скільки питань в одному запиті до AI")
    c.add_argument("--md", help="зберегти звіт у Markdown")
    c.add_argument("--json", help="зберегти результат у JSON")
    c.add_argument("--history", help="дописати рядок у JSONL-історію запусків")
    c.add_argument("--label", help="мітка запуску для історії")
    c.add_argument("--cache-dir")
    c.add_argument("--no-cache", action="store_true")
    c.set_defaults(fn=cmd_check)

    e = sub.add_parser("extract-spec", help="AI розбирає текст завдання у специфікацію JSON")
    e.add_argument("assignment")
    e.add_argument("--out", required=True)
    e.add_argument("--cache-dir")
    e.add_argument("--no-cache", action="store_true")
    e.set_defaults(fn=cmd_extract, no_ai=False)

    d = sub.add_parser("doctor", help="перевірити ключ і доступ до моделі")
    d.set_defaults(fn=cmd_doctor)

    from . import bench
    bench.add_parser(sub)
    args = ap.parse_args(argv)
    return args.fn(args)

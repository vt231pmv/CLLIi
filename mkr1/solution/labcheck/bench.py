"""Вимірювання якості самого інструмента.

  bench static — «підкинуті дефекти»: чистий синтетичний репозиторій + N видів дефектів × кілька варіантів;
                 міряємо recall (знайдено підкинуте), зайві спрацювання на чистому і на мутантах. Без AI, відтворювано.
  bench ai     — (потрібен ключ) якість AI-кроків: 1) розбір завдання проти еталонних специфікацій;
                 2) суддівство відповідей на реальних відповідях лаби 6 + підкинуті підміни.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import ai_steps
from .llm import LLMAuthError, LLMClient, LLMUnavailable, load_dotenv
from .spec import load_spec
from .static_checks import ERR, INFO, WARN, run_static

SOLUTION = Path(__file__).resolve().parents[1]
FILLER = ("Тестова відповідь на питання {n}: у цьому розділі наведено міркування, які пояснюють результат запуску, "
          "порівнюють отримані значення між собою та описують, що саме змінилося після виконаного експерименту в ноутбуці.")


# ============================================================ синтетичний чистий репозиторій
def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def _code(src, i):
    return {"cell_type": "code", "metadata": {}, "execution_count": i, "source": src.splitlines(True),
            "outputs": [{"output_type": "stream", "name": "stdout", "text": [f"результат {i}\n"]}]}


def _nb(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"kernelspec": {"name": "python3"}, "language_info": {"name": "python"},
                         "widgets": {"application/vnd.jupyter.widget-state+json": {"state": {}, "version_major": 2, "version_minor": 0}}}}


def make_clean_repo(dst: Path, spec: dict) -> None:
    files = [r["path"] for r in spec["required"] if r.get("type", "file") == "file" and r["path"].endswith(".ipynb")]
    for rel in files:
        cells, k = [_md(f"# {Path(rel).stem}\n\nВступ до ноутбука.")], 0
        for q in (q for q in spec["questions"] if q.get("file") == rel):
            k += 1
            cells += [_md(f"### Питання {q['n']}. {q['text']}"), _md(FILLER.format(n=q["n"])), _code(f"print({q['n']})", k)]
        for s in (s for s in spec["sections"] if s.get("file") == rel):
            cells.append(_md(f"## {s['title']}\n\nТекст розділу про {s['title'].lower()} із конкретними висновками."))
        k += 1
        cells.append(_code("x = 1\nprint(x)", k))
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_nb(cells), ensure_ascii=False), encoding="utf-8")


def _edit(path: Path, fn) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def _any_nb(repo: Path, rng: random.Random, pick=None) -> Path:
    nbs = sorted(repo.glob("lab6/*.ipynb"))
    return pick(nbs) if pick else rng.choice(nbs)


def _qcells(d):
    return [i for i, c in enumerate(d["cells"]) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("### Питання")]


# ============================================================ мутації
PH = ["⏳ ПІСЛЯ ЗАПУСКУ: тут буде висновок", "TODO дописати відповідь", "<ваша відповідь>", "Корпус і чому потрібен RAG: ...", "ПІСЛЯ ЗАПУСКУ"]
FAKE = [lambda r: "AIza" + "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789_-") for _ in range(35)),
        lambda r: "sk-" + "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(32)),
        lambda r: "hf_" + "".join(r.choice("abcdefghijklmnopqrstuvwxyzABCDEF0123456789") for _ in range(34)),
        lambda r: "ghp_" + "".join(r.choice("abcdefghijklmnopqrstuvwxyzABCDEF0123456789") for _ in range(36))]


def m_delete(repo, rng):
    _any_nb(repo, rng).unlink()


def m_dir_case(repo, rng):
    (repo / "lab6").rename(repo / "Lab6")


def m_copy_suffix(repo, rng):
    p = _any_nb(repo, rng)
    p.rename(p.with_name(p.stem + " (1)" + p.suffix))


def m_placeholder(repo, rng):
    def f(d):
        i = rng.choice([i for i, c in enumerate(d["cells"]) if c["cell_type"] == "markdown"])
        d["cells"][i]["source"] = d["cells"][i]["source"] + ["\n", rng.choice(PH)]
    _edit(_any_nb(repo, rng), f)


def m_drop_heading(repo, rng):
    def f(d):
        i = rng.choice(_qcells(d))
        d["cells"][i]["source"] = ["Просто текст без заголовка відповіді"]
    _edit(_any_nb(repo, rng, lambda ns: rng.choice([n for n in ns if "rag_project" not in n.name])), f)


def m_empty_answer(repo, rng):
    def f(d):
        i = rng.choice(_qcells(d))
        del d["cells"][i + 1:i + 3]
    _edit(_any_nb(repo, rng, lambda ns: rng.choice([n for n in ns if "rag_project" not in n.name])), f)


def m_secret(repo, rng):
    key = rng.choice(FAKE)(rng)
    def f(d):
        c = [c for c in d["cells"] if c["cell_type"] == "code"][0]
        c["source"] = [f'API_KEY = "{key}"\n'] + c["source"]
    _edit(_any_nb(repo, rng), f)


def m_env(repo, rng):
    (repo / "lab6" / ".env").write_text(f"LLM_API_KEY={rng.choice(FAKE)(rng)}\n", encoding="utf-8")


def m_weights(repo, rng):
    d = repo / "lab6" / "model"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"\0" * 1024)


def m_index(repo, rng):
    d = repo / "lab6" / "faiss_index"
    d.mkdir()
    (d / "index.faiss").write_bytes(b"\0" * 1024)


def m_big(repo, rng):
    with (repo / "lab6" / "corpus_dump.txt").open("wb") as fh:
        fh.truncate(101 * 2**20)  # розріджений файл: місця на диску майже не займає


def m_bad_json(repo, rng):
    p = _any_nb(repo, rng)
    p.write_text(p.read_text(encoding="utf-8")[:-40], encoding="utf-8")


def m_error_output(repo, rng):
    def f(d):
        c = [c for c in d["cells"] if c["cell_type"] == "code"][0]
        c["outputs"] = [{"output_type": "error", "ename": "ValueError", "evalue": "boom", "traceback": []}]
    _edit(_any_nb(repo, rng), f)


def m_not_run(repo, rng):
    def f(d):
        for c in d["cells"]:
            if c["cell_type"] == "code":
                c["execution_count"], c["outputs"] = None, []
    _edit(_any_nb(repo, rng), f)


def m_widgets(repo, rng):
    def f(d):
        d["metadata"]["widgets"]["application/vnd.jupyter.widget-state+json"].pop("state")
    _edit(_any_nb(repo, rng), f)


def m_junk(repo, rng):
    d = repo / "lab6" / ".ipynb_checkpoints"
    d.mkdir()
    (d / "x-checkpoint.ipynb").write_text("{}", encoding="utf-8")


def m_section(repo, rng):
    def f(d):
        i = rng.choice([i for i, c in enumerate(d["cells"]) if "".join(c["source"]).startswith("## ")])
        d["cells"][i]["source"] = ["## Щось зовсім інше\n\nбез потрібної назви"]
    _edit(repo / "lab6" / "rag_project.ipynb", f)


# name, очікуваний код, дозволені побічні коди (каскад), функція
MUTATIONS = [
    ("видалено обов'язковий файл", "E_MISSING", {"W_Q_SKIPPED", "W_SECTION"}, m_delete),
    ("регістр каталогу Lab6 замість lab6", "E_MISSING", {"W_Q_SKIPPED"}, m_dir_case),
    ("суфікс « (1)» після завантаження", "E_MISSING", {"W_Q_SKIPPED", "W_SECTION"}, m_copy_suffix),
    ("плейсхолдер у markdown", "E_PLACEHOLDER", set(), m_placeholder),
    ("зник заголовок відповіді", "E_Q_MISSING", set(), m_drop_heading),
    ("порожня відповідь", "W_Q_THIN", set(), m_empty_answer),
    ("ключ API у коді", "E_SECRET", set(), m_secret),
    ("файл .env", "E_SECRET", set(), m_env),
    ("ваги моделі", "E_FORBIDDEN", set(), m_weights),
    ("векторний індекс", "E_FORBIDDEN", set(), m_index),
    ("файл понад 100 МБ", "E_TOO_BIG", set(), m_big),
    ("пошкоджений JSON ноутбука", "E_BAD_JSON", {"W_Q_SKIPPED", "W_SECTION"}, m_bad_json),
    ("помилка у виводі клітинки", "E_CELL_ERROR", set(), m_error_output),
    ("клітинки не запускались", "W_NOT_EXECUTED", set(), m_not_run),
    ("widgets без state (GitHub не відкриє)", "W_WIDGETS_STATE", set(), m_widgets),
    ("службова тека .ipynb_checkpoints", "W_JUNK", set(), m_junk),
    ("зникла секція звіту", "W_SECTION", set(), m_section),
]


def _bad(res):
    return [f for f in res.findings if f.sev in (ERR, WARN)]


def run_static_bench(variants: int = 5, spec_path: Path | None = None) -> dict:
    spec = load_spec(spec_path or SOLUTION / "specs" / "lab6.json")
    spec = {**spec, "deadline": None}
    rows, fp_clean, tp_findings, all_findings = [], 0, 0, 0
    times: list[float] = []
    for v in range(variants):  # зайві спрацювання на ЧИСТОМУ репозиторії
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t)
            make_clean_repo(repo, spec)
            t0 = time.perf_counter()
            res = run_static(repo, spec)
            times.append(time.perf_counter() - t0)
            fp_clean += len(_bad(res))
    for name, code, extra, fn in MUTATIONS:
        hit = 0
        for v in range(variants):
            rng = random.Random(f"{name}-{v}")
            with tempfile.TemporaryDirectory() as t:
                repo = Path(t)
                make_clean_repo(repo, spec)
                fn(repo, rng)
                res = run_static(repo, spec)
            bad = _bad(res)
            if any(f.code == code for f in bad):
                hit += 1
            all_findings += len(bad)
            tp_findings += sum(1 for f in bad if f.code == code or f.code in extra)
        rows.append({"defect": name, "expected": code, "found": hit, "variants": variants})
    total, found = sum(r["variants"] for r in rows), sum(r["found"] for r in rows)
    return {"kinds": len(MUTATIONS), "variants_per_kind": variants, "mutants": total, "found": found,
            "recall": round(found / total, 4), "findings_on_clean": fp_clean, "clean_runs": variants,
            "precision_of_findings": round(tp_findings / all_findings, 4) if all_findings else 1.0,
            "mean_run_s_clean_synthetic": round(sum(times) / len(times), 4), "rows": rows}


# ============================================================ порівняння специфікацій (для AI-розбору)
def _ratio(a, b):
    import difflib
    return difflib.SequenceMatcher(None, ai_steps.norm(a), ai_steps.norm(b)).ratio()


def _pn(p):
    return ai_steps.norm_path(p) if p else None


def compare_specs(got: dict, gold: dict) -> dict:
    gp, tp = {ai_steps.norm_path(r["path"]) for r in got["required"]}, {ai_steps.norm_path(r["path"]) for r in gold["required"]}
    inter = len(gp & tp)
    gq = {q["n"]: q for q in got["questions"]}
    text_ok = file_ok = 0
    for q in gold["questions"]:
        g = gq.get(q["n"])
        if g and _ratio(g["text"], q["text"]) >= 0.95:
            text_ok += 1
            file_ok += int(_pn(g.get("file")) == _pn(q.get("file")))
    sec_miss = [s["title"] for s in gold["sections"] if not any(_ratio(s["title"], g["title"]) >= 0.8 for g in got["sections"])]
    sec_ok = len(gold["sections"]) - len(sec_miss)
    nq = len(gold["questions"])
    return {"required_precision": round(inter / len(gp), 4) if gp else 0.0, "required_recall": round(inter / len(tp), 4),
            "questions_extracted": len(got["questions"]), "questions_gold": nq,
            "question_text_ok": round(text_ok / nq, 4), "question_file_ok": round(file_ok / nq, 4),
            "sections_recall": round(sec_ok / len(gold["sections"]), 4) if gold["sections"] else 1.0, "sections_missing": sec_miss,
            "forbidden_ok": {f["kind"] for f in got["forbidden"]} == {f["kind"] for f in gold["forbidden"]},
            "deadline_ok": got.get("deadline") == gold.get("deadline"),
            "media_ok": [(m["min_s"], m["max_s"]) for m in got.get("media", [])] == [(m["min_s"], m["max_s"]) for m in gold.get("media", [])]}


def run_ai_spec_bench(llm: LLMClient, names=("lab5", "lab6", "mkr1"), passes: int = 3, pause_s: float = 60, sleep=time.sleep) -> dict:
    """Розбір завдань AI → порівняння з еталоном. Якщо сервіс недоступний (503/429 навіть після повторів клієнта),
    завдання пробуємо ще раз після паузи — до `passes` проходів. Помилки формату (ValueError) не повторюємо: це не збій сервісу."""
    out: dict[str, dict] = {}
    tok = {n: {"prompt_tokens": 0, "completion_tokens": 0} for n in names}
    pending = list(names)
    for attempt in range(1, passes + 1):
        still = []
        for n in pending:
            text = (SOLUTION / "inputs" / f"{n}_assignment.txt").read_text(encoding="utf-8")
            gold = load_spec(SOLUTION / "specs" / f"{n}.json")
            before = llm.stats.as_dict()
            t0 = time.perf_counter()
            try:
                spec, st = ai_steps.extract_spec(llm, text)
                out[n] = {"ok": True, "latency_s": round(time.perf_counter() - t0, 2), "verification": st, **compare_specs(spec, gold)}
            except LLMUnavailable as e:
                out[n] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                if not isinstance(e, LLMAuthError):
                    still.append(n)
            except ValueError as e:
                out[n] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            after = llm.stats.as_dict()
            for k in ("prompt_tokens", "completion_tokens"):
                tok[n][k] += after[k] - before[k]
            out[n]["tokens"] = dict(tok[n])
            if attempt > 1:
                out[n]["attempts"] = attempt
        pending = still
        if not pending:
            break
        if attempt < passes:
            print(f"  сервіс недоступний для: {', '.join(pending)}; пауза {pause_s:.0f} с і ще одна спроба ({attempt + 1}/{passes})...", flush=True)
            sleep(pause_s)
    return out


# ============================================================ суддівство: реальні відповіді + підкинуті підміни
NONANSWER = ("Це питання потребує уважного розгляду, і до нього варто повернутися пізніше. Відповідь буде додано після того, як "
             "будуть проаналізовані всі матеріали лабораторної роботи та зібрано додаткові спостереження.")


def build_judge_items(data: list[dict]) -> list[dict]:
    """Оригінали (мають бути answered) + 10 підмін відповіді іншого питання + 10 відписок без змісту."""
    by_n = {d["n"]: d for d in data}
    items = [{"n": d["n"], "kind": "original", "question": d["question"], "answer": d["answer"]} for d in data]
    ns = sorted(by_n)
    for k, n in enumerate(ns):
        if k % 3 == 2:   # 10 підмін
            other = ns[(k + 7) % len(ns)]
            items.append({"n": n, "kind": "swapped", "question": by_n[n]["question"], "answer": by_n[other]["answer"]})
        if k % 3 == 0:   # 10 відписок
            items.append({"n": n, "kind": "filler", "question": by_n[n]["question"], "answer": NONANSWER})
    return items


def run_ai_judge_bench(llm: LLMClient, data_path: Path | None = None, batch: int = 5) -> dict:
    from .static_checks import QuestionStatus
    data = json.loads((data_path or SOLUTION / "bench_data" / "lab6_answers.json").read_text(encoding="utf-8"))
    items = build_judge_items(data["items"])
    sts = []
    for k, it in enumerate(items):
        # номери в межах одного запиту мають бути унікальними — використовуємо наскрізний індекс
        s = QuestionStatus(k + 1, "bench", it["question"], found=True, words=len(it["answer"].split()), code_cells=1,
                           block_text="### заголовок\n" + it["answer"])
        sts.append(s)
    before = llm.stats.as_dict()
    t0 = time.perf_counter()
    cnt = ai_steps.judge_questions(llm, sts, batch=batch, max_calls=60)
    dt = time.perf_counter() - t0
    per_kind: dict[str, dict] = {}
    for it, s in zip(items, sts):
        d = per_kind.setdefault(it["kind"], {"n": 0, "answered": 0, "partial": 0, "off_topic": 0, "missing": 0, "needs_review": 0})
        d["n"] += 1
        d[s.verdict] = d.get(s.verdict, 0) + 1
    o, sw, fl = per_kind["original"], per_kind["swapped"], per_kind["filler"]
    flagged = lambda d: d["n"] - d["answered"]
    answered_q = sum(1 for s in sts if s.verdict in ("answered", "partial"))
    after = llm.stats.as_dict()
    return {
        "items": len(items), "batches": cnt["calls"], "failed_batches": cnt["failed_batches"],
        "originals_answered_rate": round(o["answered"] / o["n"], 4),
        "originals_false_alarm_rate": round((o["off_topic"] + o["missing"]) / o["n"], 4),
        "swapped_caught_rate": round(flagged(sw) / sw["n"], 4), "filler_caught_rate": round(flagged(fl) / fl["n"], 4),
        "defects_missed": sw["answered"] + fl["answered"],
        "quote_rejected": sum(1 for s in sts if s.verdict == "needs_review"), "accepted_with_quote": answered_q,
        "per_kind": per_kind, "latency_s": round(dt, 2),
        # чистий час запитів (без пауз між повторами при 503/429) — саме він іде в оцінку «часу після»; latency_s — час на стіні
        "request_latency_s": round(after["latency_s"] - before["latency_s"], 2),
        "tokens": {k: after[k] - before[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd", "calls", "cache_hits")},
        "verdicts": [{"kind": it["kind"], "n": it["n"], "verdict": s.verdict, "reason": s.reason,
                     **({"rejected_quote": s.rejected_quote} if s.rejected_quote else {})} for it, s in zip(items, sts)],
    }


# ============================================================ CLI
def _print_static(r: dict) -> None:
    print(f"Підкинуті дефекти: {r['kinds']} видів × {r['variants_per_kind']} варіантів = {r['mutants']} мутантів")
    for row in r["rows"]:
        print(f"  {'OK ' if row['found'] == row['variants'] else 'ПРОПУСК'} {row['defect']:<42} {row['expected']:<16} {row['found']}/{row['variants']}")
    print(f"recall: {r['found']}/{r['mutants']} = {r['recall']:.1%}")
    print(f"зайвих спрацювань на чистому репозиторії: {r['findings_on_clean']} (у {r['clean_runs']} запусках)")
    print(f"частка знахідок, що відповідають підкинутому дефекту або його каскаду: {r['precision_of_findings']:.1%}")
    print(f"середній час перевірки синтетичного репозиторію: {r['mean_run_s_clean_synthetic']:.3f} с")


def cmd_bench(args) -> int:
    if args.what == "static":
        r = run_static_bench(args.variants)
        _print_static(r)
        if args.json:
            Path(args.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0 if r["recall"] == 1.0 and r["findings_on_clean"] == 0 else 1
    load_dotenv()
    llm = LLMClient.from_env(cache_dir=None if args.no_cache else Path(args.cache_dir or SOLUTION / ".labcheck_cache"))
    if llm is None:
        print("Потрібні LLM_API_KEY і LLM_MODEL (див. .env.example).", file=sys.stderr)
        return 2
    res = {"model": llm.model}
    if args.only in ("all", "spec"):
        res["spec"] = run_ai_spec_bench(llm)
        for n, v in res["spec"].items():
            print(f"[spec/{n}]", json.dumps({k: v[k] for k in v if k not in ("verification", "tokens")}, ensure_ascii=False))
    if args.only in ("all", "judge"):
        res["judge"] = run_ai_judge_bench(llm)
        print("[judge]", json.dumps({k: v for k, v in res["judge"].items() if k not in ("verdicts", "per_kind")}, ensure_ascii=False))
    res["llm_total"] = llm.stats.as_dict()
    print("[токени/вартість]", json.dumps(res["llm_total"], ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def add_parser(sub) -> None:
    b = sub.add_parser("bench", help="виміряти якість інструмента")
    b.add_argument("what", choices=["static", "ai"])
    b.add_argument("--variants", type=int, default=5)
    b.add_argument("--only", choices=["all", "spec", "judge"], default="all")
    b.add_argument("--json")
    b.add_argument("--cache-dir")
    b.add_argument("--no-cache", action="store_true")
    b.set_defaults(fn=cmd_bench)

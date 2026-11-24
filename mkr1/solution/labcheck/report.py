"""Формування результату: короткий текст для термінала, Markdown-звіт, JSON і рядок історії запусків."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .static_checks import ERR, INFO, WARN, StaticResult

MARK = {ERR: "[ERR ]", WARN: "[WARN]", INFO: "[info]"}
VERDICT_UA = {"answered": "є відповідь", "partial": "неповна", "off_topic": "не за темою", "missing": "немає відповіді",
              "needs_review": "перевірити вручну", "": "не оцінювалось (без AI)"}


@dataclass
class CheckReport:
    spec: dict
    spec_source: str
    res: StaticResult
    duration_s: float
    spec_stats: dict | None = None
    judge: dict | None = None
    llm_stats: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.res.count(ERR)

    @property
    def warnings(self) -> int:
        n = self.res.count(WARN)
        n += sum(1 for q in self.res.questions if q.verdict in ("partial", "off_topic", "missing", "needs_review"))
        return n

    @property
    def questions_found(self) -> tuple[int, int]:
        return sum(q.found for q in self.res.questions), len(self.res.questions)


def ai_warnings(rep: CheckReport) -> list[tuple[str, str, str]]:
    out = []
    for q in rep.res.questions:
        if q.found and q.verdict in ("partial", "off_topic", "missing", "needs_review"):
            extra = f" Модель цитувала: «{q.rejected_quote[:120]}»." if q.rejected_quote else ""
            out.append((f"W_AI_{q.verdict.upper()}", f"{q.file}", f"питання {q.n}: {VERDICT_UA[q.verdict]} — {q.reason}{extra}"))
    return out


def render_text(rep: CheckReport) -> str:
    r, L = rep.res, []
    L.append(f"labcheck | {rep.spec.get('title')} | специфікація: {rep.spec_source}")
    L.append(f"тека: {r.base}")
    L.append("")
    L.append("--- Обов'язкові файли ---")
    for path, status in r.files:
        L.append(f"{'[ OK ]' if status == 'є' else '[FAIL]'} {path}" + ("" if status == 'є' else f"  ({status})"))
    L.append("")
    L.append("--- Знахідки ---")
    shown = [f for f in r.findings if f.sev != INFO] + [type("F", (), {"sev": WARN, "code": c, "where": w, "msg": m}) for c, w, m in ai_warnings(rep)]
    if not shown:
        L.append("(немає)")
    for f in shown:
        L.append(f"{MARK[f.sev]} {f.code} {f.where}: {f.msg}")
    for f in r.findings:
        if f.sev == INFO:
            L.append(f"{MARK[INFO]} {f.code} {f.where}: {f.msg}")
    if r.questions:
        a, b = rep.questions_found
        L.append("")
        L.append("--- Питання ---")
        L.append(f"знайдено заголовків відповідей: {a} із {b}")
        if any(q.verdict for q in r.questions):
            tally: dict[str, int] = {}
            for q in r.questions:
                if q.found:
                    tally[q.verdict or "—"] = tally.get(q.verdict or "—", 0) + 1
            L.append("вердикти AI: " + ", ".join(f"{VERDICT_UA.get(k, k)} — {v}" for k, v in sorted(tally.items())))
    L.append("")
    L.append("--- Підсумок ---")
    extra = ""
    if rep.llm_stats:
        s = rep.llm_stats
        extra = f" | AI: {s['calls']} викл., {s['cache_hits']} з кешу, {s['prompt_tokens']}+{s['completion_tokens']} токенів, ${s['cost_usd']:.4f}"
    L.append(f"помилок: {rep.errors} | попереджень: {rep.warnings} | час: {rep.duration_s:.2f} с{extra}")
    for n in rep.notes:
        L.append(f"примітка: {n}")
    if rep.errors:
        L.append("ВИСНОВОК: НЕ ГОТОВО до здачі — виправте помилки [ERR].")
    elif rep.warnings:
        L.append("ВИСНОВОК: помилок немає; перегляньте попередження [WARN] перед здачею.")
    else:
        L.append("ВИСНОВОК: ГОТОВО до здачі.")
    return "\n".join(L) + "\n"


def render_md(rep: CheckReport) -> str:
    r = rep.res
    L = [f"# Звіт перевірки: {rep.spec.get('title')}", "",
         f"- специфікація: `{rep.spec_source}`", f"- тека: `{r.base}`",
         f"- помилок: **{rep.errors}**, попереджень: **{rep.warnings}**, час: {rep.duration_s:.2f} с", ""]
    L += ["## Файли", "", "| Шлях | Статус |", "|---|---|"]
    L += [f"| `{p}` | {s} |" for p, s in r.files]
    L += ["", "## Знахідки", "", "| Рівень | Код | Де | Опис |", "|---|---|---|---|"]
    rows = [(f.sev, f.code, f.where, f.msg) for f in r.findings] + [(WARN, c, w, m) for c, w, m in ai_warnings(rep)]
    L += [f"| {s} | {c} | `{w}` | {m} |" for s, c, w, m in rows] or ["| — | — | — | знахідок немає |"]
    if r.questions:
        L += ["", "## Питання", "", "| № | Файл | Знайдено | Слів | Вердикт AI |", "|---|---|---|---|---|"]
        L += [f"| {q.n} | `{q.file}` | {'так' if q.found else 'НІ'} | {q.words} | {VERDICT_UA.get(q.verdict, q.verdict)} |" for q in r.questions]
    return "\n".join(L) + "\n"


def to_json(rep: CheckReport) -> dict:
    return {
        "title": rep.spec.get("title"), "spec_source": rep.spec_source, "base": str(rep.res.base),
        "errors": rep.errors, "warnings": rep.warnings, "duration_s": round(rep.duration_s, 3),
        "files": [{"path": p, "status": s} for p, s in rep.res.files],
        "findings": [f.as_dict() for f in rep.res.findings],
        "questions": [{"n": q.n, "file": q.file, "found": q.found, "words": q.words, "verdict": q.verdict, "reason": q.reason} for q in rep.res.questions],
        "spec_stats": rep.spec_stats, "judge": rep.judge, "llm": rep.llm_stats,
    }


def history_line(rep: CheckReport, label: str = "") -> str:
    a, b = rep.questions_found
    d = {"ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "label": label,
         "title": rep.spec.get("title"), "target": rep.res.base.name, "errors": rep.errors, "warnings": rep.warnings,
         "questions_found": f"{a}/{b}", "duration_s": round(rep.duration_s, 3), "ai": bool(rep.llm_stats and (rep.llm_stats["calls"] or rep.llm_stats["cache_hits"])),
         "llm": rep.llm_stats, "os": platform.system()}
    return json.dumps(d, ensure_ascii=False)

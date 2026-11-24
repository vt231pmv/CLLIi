"""Специфікація завдання (чекліст): що має лежати в репозиторії і на які питання потрібні відповіді.

Формат JSON (див. specs/*.json):
  title, deadline (YYYY-MM-DD | null),
  required  [{path, type: file|dir, quote}],
  forbidden [{kind: vector_index|model_weights, quote}],
  media     [{path, min_s, max_s, quote}]            (необов'язково),
  questions [{n, text, file}],
  sections  [{title, file}]
"""
from __future__ import annotations

import json
from pathlib import Path

FORBIDDEN_KINDS = {"vector_index", "model_weights"}


class SpecError(ValueError):
    pass


def validate_spec(d: object) -> list[str]:
    """Повертає список проблем; порожній список = специфікація придатна."""
    problems: list[str] = []
    if not isinstance(d, dict):
        return ["специфікація має бути JSON-об'єктом"]
    req = d.get("required")
    if not isinstance(req, list) or not req:
        problems.append("required: потрібен непорожній список")
    else:
        for i, r in enumerate(req):
            if not isinstance(r, dict) or not isinstance(r.get("path"), str) or not r["path"].strip():
                problems.append(f"required[{i}]: немає path")
            elif r.get("type", "file") not in ("file", "dir"):
                problems.append(f"required[{i}]: type має бути file або dir")
    for i, f in enumerate(d.get("forbidden", []) or []):
        if not isinstance(f, dict) or f.get("kind") not in FORBIDDEN_KINDS:
            problems.append(f"forbidden[{i}]: kind має бути одним із {sorted(FORBIDDEN_KINDS)}")
    seen = set()
    for i, q in enumerate(d.get("questions", []) or []):
        if not isinstance(q, dict) or not isinstance(q.get("n"), int) or not isinstance(q.get("text"), str):
            problems.append(f"questions[{i}]: потрібні n (int) і text (str)")
            continue
        if q["n"] in seen:
            problems.append(f"questions[{i}]: дубль номера {q['n']}")
        seen.add(q["n"])
    for i, s in enumerate(d.get("sections", []) or []):
        if not isinstance(s, dict) or not isinstance(s.get("title"), str):
            problems.append(f"sections[{i}]: потрібен title")
    return problems


def load_spec(path: str | Path) -> dict:
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as e:
        raise SpecError(f"немає файлу специфікації: {p}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"специфікація {p} не є валідним JSON: {e}") from e
    problems = validate_spec(d)
    if problems:
        raise SpecError(f"специфікація {p} некоректна: " + "; ".join(problems[:5]))
    d.setdefault("questions", [])
    d.setdefault("sections", [])
    d.setdefault("forbidden", [])
    d.setdefault("media", [])
    d.setdefault("deadline", None)
    d.setdefault("title", p.stem)
    return d


def top_dirs(spec: dict) -> list[str]:
    """Верхні каталоги зі шляхів вимог (lab6, mkr1 ...): саме їх вміст перевіряємо на зайві файли."""
    out: list[str] = []
    for r in spec["required"]:
        parts = Path(r["path"]).parts
        if len(parts) > 1 and parts[0] not in out:
            out.append(parts[0])
    return out

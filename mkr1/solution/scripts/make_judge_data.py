"""Збирає дані для бенчмарку AI-судді з реальних відповідей лаби 6 (30 питань): питання + текст відповіді з ноутбука.

Запуск: python scripts/make_judge_data.py <тека_з_ноутбуками_lab6> [вихід.json]
(Відповіді — власний текст студента з його ж репозиторію, тому додавати їх сюди можна; чужих даних немає.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labcheck.notebook import find_blocks, load_notebook  # noqa: E402
from labcheck.spec import load_spec  # noqa: E402


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "bench_data" / "lab6_answers.json"
    spec = load_spec(ROOT / "specs" / "lab6.json")
    items = []
    for rel in sorted({q["file"] for q in spec["questions"]}):
        blocks = find_blocks(load_notebook(src / Path(rel).name))
        for q in (q for q in spec["questions"] if q["file"] == rel):
            t = blocks[q["n"]].text()
            items.append({"n": q["n"], "question": q["text"], "answer": t.split("\n", 1)[1].strip()})
    items.sort(key=lambda x: x["n"])
    out.write_text(json.dumps({"source": "реальні відповіді студента в ноутбуках лаби 6 (версія 24.11.2026)", "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} відповідей → {out} ({out.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()

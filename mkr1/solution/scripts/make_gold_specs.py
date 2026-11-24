"""Будує ЕТАЛОННІ специфікації завдань детермінованим парсером (без AI).

Еталони потрібні для двох речей:
  1) запасний варіант, коли AI-розбір недоступний (fallback);
  2) перевірка якості AI-розбору: порівнюємо те, що витягла модель, з еталоном (bench ai).

Запуск:  python scripts/make_gold_specs.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[str]:
    return (ROOT / "inputs" / name).read_text(encoding="utf-8").splitlines()


def between(lines: list[str], start_re: str, end_re: str) -> list[str]:
    s = next(i for i, l in enumerate(lines) if re.match(start_re, l))
    e = next(i for i, l in enumerate(lines) if i > s and re.match(end_re, l))
    return [l.strip() for l in lines[s + 1:e] if l.strip()]


def lab_spec(name, title, deadline, folder, nb_files, ranges, start_q, end_q, rep_start, forb):
    lines = read(f"{name}_assignment.txt")
    qs = between(lines, start_q, end_q)
    assert qs, name
    file_for = {}
    for (a, b), f in zip(ranges, nb_files[:3]):
        for n in range(a, b + 1):
            file_for[n] = f"{folder}/{f}"
    assert len(qs) == max(file_for), (name, len(qs), max(file_for))
    sections = []
    for l in between(lines, rep_start, r"^Як здавати"):
        title_part = l.split(":")[0].strip()
        if title_part:
            sections.append({"title": title_part, "file": f"{folder}/{nb_files[3]}"})
    required = [{"path": f"{folder}/{f}", "type": "file", "quote": f"/{folder}/{f}"} for f in nb_files]
    return {
        "title": title, "deadline": deadline, "forbidden": forb,
        "required": required,
        "questions": [{"n": i + 1, "text": q, "file": file_for[i + 1]} for i, q in enumerate(qs)],
        "sections": sections,
    }


def main():
    lab6 = lab_spec(
        "lab6", "Лабораторна 6. Генерація, доповнена пошуком (RAG)", "2026-12-12", "lab6",
        ["01_embeddings.ipynb", "02_rag_from_scratch.ipynb", "03_rag_langchain_eval.ipynb", "rag_project.ipynb"],
        [(1, 10), (11, 20), (21, 30)], r"^0\.1\.", r"^1\. Власна RAG", r"^1\.8\.",
        [{"kind": "vector_index", "quote": "Векторний індекс і ваги моделей у репозиторій не додавайте"},
         {"kind": "model_weights", "quote": "Векторний індекс і ваги моделей у репозиторій не додавайте"}],
    )
    lab5 = lab_spec(
        "lab5", "Лабораторна 5. Донавчання моделей (Fine-Tuning)", "2026-11-28", "lab5",
        ["01_bert_head.ipynb", "02_chat_templates.ipynb", "03_sft_lora.ipynb", "ft_project.ipynb"],
        [(1, 10), (11, 21), (22, 30)], r"^0\.1\.", r"^1\. Власне донавчання", r"^1\.6\.",
        [{"kind": "model_weights", "quote": "Ваги моделі чи адаптера в репозиторій не додавайте"}],
    )
    lines = read("mkr1_assignment.txt")
    qs = between(lines, r"^У звіті дайте відповіді", r"^Результат роботи")
    assert len(qs) == 12, len(qs)
    mkr = {
        "title": "МКР-1. Як я покращив/ла своє життя за допомогою AI", "deadline": "2026-12-01",
        "forbidden": [],
        "required": [
            {"path": "mkr1/report.md", "type": "file", "quote": "/mkr1/report.md"},
            {"path": "mkr1/solution", "type": "dir", "quote": "/mkr1/solution/*"},
            {"path": "mkr1/demo.mp4", "type": "file", "quote": "/mkr1/demo.mp4"},
            {"path": "mkr1/evidence", "type": "dir", "quote": "/mkr1/evidence/*"},
        ],
        "media": [{"path": "mkr1/demo.mp4", "min_s": 60, "max_s": 90, "quote": "Відео 60–90 секунд"}],
        "questions": [{"n": i + 1, "text": q, "file": "mkr1/report.md"} for i, q in enumerate(qs)],
        # назви розділів звіту — дослівні пункти зі списку «Звіт» у тексті завдання
        "sections": [{"title": t, "file": "mkr1/report.md"} for t in (
            "архітектура рішення", "використані AI-технології та інструменти", "обґрунтування вибору підходу",
            "таблиця ризиків", "таблиця результатів «до / після»", "відповіді на 12 питань з п.8")],
    }
    out = ROOT / "specs"
    out.mkdir(exist_ok=True)
    for name, spec in (("lab5", lab5), ("lab6", lab6), ("mkr1", mkr)):
        (out / f"{name}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(name, "вимог до файлів:", len(spec["required"]), "питань:", len(spec["questions"]), "секцій:", len(spec["sections"]))


if __name__ == "__main__":
    main()

"""Реальні прогони інструмента на справжніх ноутбуках студента (до і після доопрацювання) з логами для evidence/.

Вхідні файли — це ноутбуки, які студент завантажив після запуску в Colab (імена в uploads мають префікс-хеш),
і фінальні версії. Скрипт збирає з них тимчасове дерево з правильними іменами, запускає labcheck і зберігає:
  evidence/real_runs/<мітка>.txt   — вивід програми (+ SHA-256 вхідних файлів для відстеження походження)
  evidence/history.jsonl           — по рядку на кожен запуск (історія)
  evidence/timing.json             — час N повторних запусків
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labcheck.cli import main  # noqa: E402

UP = Path("/mnt/user-data/uploads")
CASES = {
    "lab6_before": ("lab6", {"01_embeddings.ipynb": UP / "9a472e47-01_embeddings.ipynb", "02_rag_from_scratch.ipynb": UP / "cf6e915a-02_rag_from_scratch.ipynb",
                             "03_rag_langchain_eval.ipynb": UP / "c9c68c45-03_rag_langchain_eval.ipynb", "rag_project.ipynb": UP / "36d8a679-rag_project.ipynb"}),
    "lab6_after": ("lab6", {n: Path("/home/claude/lab6/final") / n for n in ("01_embeddings.ipynb", "02_rag_from_scratch.ipynb", "03_rag_langchain_eval.ipynb", "rag_project.ipynb")}),
    "lab5_before": ("lab5", {"01_bert_head.ipynb": UP / "431eec24-01_bert_head.ipynb", "02_chat_templates.ipynb": UP / "54c45894-02_chat_templates.ipynb",
                             "03_sft_lora.ipynb": UP / "ba85d94a-03_sft_lora.ipynb", "ft_project.ipynb": UP / "76bcd665-ft_project.ipynb"}),
    "lab5_after": ("lab5", {n: Path("/home/claude/final5") / n for n in ("01_bert_head.ipynb", "02_chat_templates.ipynb", "03_sft_lora.ipynb", "ft_project.ipynb")}),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def build(top: str, files: dict) -> Path:
    t = Path(tempfile.mkdtemp(prefix="labcheck_"))
    (t / top).mkdir()
    for name, src in files.items():
        shutil.copy(src, t / top / name)
    return t


def run_once(tree: Path, spec: str, extra: list[str]) -> tuple[int, str, float]:
    buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        code = main(["check", str(tree), "--spec", str(ROOT / "specs" / f"{spec}.json"), "--no-ai", *extra])
    return code, buf.getvalue(), time.perf_counter() - t0


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    ev = Path(a.evidence)
    (ev / "real_runs").mkdir(parents=True, exist_ok=True)
    hist = ev / "history.jsonl"
    timing = {}
    for label, (top, files) in CASES.items():
        tree = build(top, files)
        size = sum(p.stat().st_size for p in (tree / top).iterdir())
        code, out, _ = run_once(tree, top, ["--history", str(hist), "--label", label])
        head = [f"# {label}: labcheck check --spec specs/{top}.json --no-ai", f"# код завершення: {code}", f"# дата: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "# вхідні файли (SHA-256, перші 12 символів):"] + [f"#   {n}  {sha(p)}  {p.stat().st_size} Б" for n, p in files.items()] + [""]
        (ev / "real_runs" / f"{label}.txt").write_text("\n".join(head) + out, encoding="utf-8")
        ts = [run_once(tree, top, [])[2] for _ in range(a.repeats)]
        timing[label] = {"repeats": a.repeats, "mean_s": round(statistics.mean(ts), 4), "median_s": round(statistics.median(ts), 4),
                         "max_s": round(max(ts), 4), "input_mb": round(size / 2**20, 2)}
        print(f"{label}: код {code}, середній час {timing[label]['mean_s']} с, вхід {timing[label]['input_mb']} МБ")
        shutil.rmtree(tree, ignore_errors=True)
    (ev / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main_cli()

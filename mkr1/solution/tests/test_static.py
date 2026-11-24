import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from labcheck import bench
from labcheck.notebook import find_blocks, load_any, load_markdown_file, read_text_any
from labcheck.spec import SpecError, load_spec, validate_spec
from labcheck.static_checks import ERR, run_static

SPEC = Path(__file__).resolve().parents[1] / "specs" / "lab6.json"


@pytest.fixture()
def spec():
    return {**load_spec(SPEC), "deadline": None}


@pytest.fixture()
def repo(tmp_path, spec):
    bench.make_clean_repo(tmp_path, spec)
    return tmp_path


def codes(res, sev=None):
    return {f.code for f in res.findings if sev is None or f.sev == sev}


def test_clean_repo_has_no_findings(repo, spec):
    res = run_static(repo, spec)
    assert [f for f in res.findings if f.sev != "INFO"] == []
    assert sum(q.found for q in res.questions) == 30


def test_runs_from_inside_lab_folder(repo, spec):
    """Користувач запускає з теки lab6 — база визначається сама."""
    res = run_static(repo / "lab6", spec)
    assert not codes(res, ERR)


def test_missing_file_gets_near_miss_hint(repo, spec):
    p = repo / "lab6" / "rag_project.ipynb"
    p.rename(p.with_name("rag_project (1).ipynb"))
    res = run_static(repo, spec)
    f = next(f for f in res.findings if f.code == "E_MISSING")
    assert "rag_project (1).ipynb" in f.msg


def test_case_sensitivity_like_github(repo, spec):
    (repo / "lab6").rename(repo / "Lab6")
    assert "E_MISSING" in codes(run_static(repo, spec), ERR)


@pytest.mark.parametrize("text", ["⏳ ПІСЛЯ ЗАПУСКУ", "TODO", "<ваша відповідь>", "Корпус: ...", "..."])
def test_placeholders_detected(repo, spec, text):
    p = repo / "lab6" / "01_embeddings.ipynb"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cells"][0]["source"] += ["\n", text]
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert "E_PLACEHOLDER" in codes(run_static(repo, spec))


def test_normal_ellipsis_in_sentence_is_not_placeholder(repo, spec):
    p = repo / "lab6" / "01_embeddings.ipynb"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cells"][0]["source"] += ["\n", "Модель повертає вектори розмірності 1024, ... і так далі для кожного речення."]
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert "E_PLACEHOLDER" not in codes(run_static(repo, spec))


def test_secret_value_is_masked_in_output(repo, spec):
    key = "AIza" + "a1B2c3D4e5" * 3 + "xyz12"
    p = repo / "lab6" / "02_rag_from_scratch.ipynb"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cells"][-1]["source"] = [f'KEY = "{key}"']
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    res = run_static(repo, spec)
    f = next(f for f in res.findings if f.code == "E_SECRET")
    assert key not in f.msg and key not in f.where


def test_env_example_is_allowed_but_env_is_not(repo, spec):
    (repo / "lab6" / ".env.example").write_text("LLM_API_KEY=\n", encoding="utf-8")
    assert "E_SECRET" not in codes(run_static(repo, spec))
    (repo / "lab6" / ".env").write_text("LLM_API_KEY=abc\n", encoding="utf-8")
    assert "E_SECRET" in codes(run_static(repo, spec))


def test_forbidden_artifacts(repo, spec):
    (repo / "lab6" / "w").mkdir()
    (repo / "lab6" / "w" / "model.safetensors").write_bytes(b"0")
    (repo / "lab6" / "faiss_index").mkdir()
    res = run_static(repo, spec)
    assert sum(1 for f in res.findings if f.code == "E_FORBIDDEN") == 2


def test_broken_json_reports_instead_of_crashing(repo, spec):
    p = repo / "lab6" / "03_rag_langchain_eval.ipynb"
    p.write_text("{ not json", encoding="utf-8")
    res = run_static(repo, spec)
    assert "E_BAD_JSON" in codes(res)


def test_cp1251_encoded_markdown_is_read(tmp_path):
    p = tmp_path / "report.md"
    p.write_bytes("### Питання 1. Яка проблема?\nПроблема в тому, що перевірка забирає багато часу щоразу перед здачею роботи.".encode("cp1251"))
    nb = load_markdown_file(p)
    assert 1 in find_blocks(nb)


def test_local_numbering_is_mapped(repo, spec):
    p = repo / "lab6" / "02_rag_from_scratch.ipynb"
    d = json.loads(p.read_text(encoding="utf-8"))
    for c in d["cells"]:
        s = "".join(c["source"])
        if s.startswith("### Питання "):
            n = int(s.split()[2].rstrip("."))
            c["source"] = [s.replace(f"Питання {n}.", f"Питання {n - 10}.", 1)]
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    res = run_static(repo, spec)
    assert "E_Q_MISSING" not in codes(res) and "I_LOCAL_NUMBERING" in codes(res)


def test_numbered_demo_sections_are_not_questions(tmp_path):
    p = tmp_path / "x.ipynb"
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["### 1. Модель ембедингів"]},
             {"cell_type": "markdown", "metadata": {}, "source": ["### Питання 3. Що?"]}]
    p.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 0}), encoding="utf-8")
    assert sorted(find_blocks(load_any(p))) == [3]


def test_markdown_report_bare_number_headings(tmp_path):
    p = tmp_path / "report.md"
    p.write_text("## 8. Відповіді\n\n### 1. Яку проблему?\nТекст.\n\n### 2. Чому?\nТекст.\n", encoding="utf-8")
    assert sorted(find_blocks(load_any(p))) == [1, 2]


def test_spec_validation():
    assert validate_spec({"required": []})
    assert validate_spec({"required": [{"path": "a"}], "questions": [{"n": 1, "text": "x"}, {"n": 1, "text": "y"}]})
    with pytest.raises(SpecError):
        load_spec("нема_такого.json")


def test_static_benchmark_is_perfect():
    r = bench.run_static_bench(variants=2)
    assert r["recall"] == 1.0 and r["findings_on_clean"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="немає ffmpeg")
def test_demo_length_check(tmp_path):
    spec = {"title": "t", "required": [{"path": "mkr1/demo.mp4", "type": "file"}], "media": [{"path": "mkr1/demo.mp4", "min_s": 60, "max_s": 90}],
            "questions": [], "sections": [], "forbidden": []}
    (tmp_path / "mkr1").mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "3", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", str(tmp_path / "mkr1" / "demo.mp4")], check=True)
    res = run_static(tmp_path, spec)
    assert "E_MEDIA_LEN" in codes(res)  # 3 с — менше за 60


def test_gitignored_secret_file_and_junk_are_not_errors(repo, spec):
    (repo / "lab6" / ".gitignore").write_text(".env\n__pycache__/\n", encoding="utf-8")
    (repo / "lab6" / ".env").write_text("LLM_API_KEY=abc\n", encoding="utf-8")
    (repo / "lab6" / "__pycache__").mkdir()
    res = run_static(repo, spec)
    assert not [f for f in res.findings if f.sev in ("ERR", "WARN")]
    assert "I_GITIGNORED" in codes(res)


def test_relative_dot_root_is_resolved(repo, spec, monkeypatch):
    """Запуск `check ..` з підтеки: тека називається lab6, але Path('..').name порожнє."""
    monkeypatch.chdir(repo / "lab6")
    res = run_static(Path("."), spec)
    assert not codes(res, ERR)

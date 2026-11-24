import json
from pathlib import Path

from labcheck import bench
from labcheck.cli import main
from labcheck.spec import load_spec

SPEC = Path(__file__).resolve().parents[1] / "specs" / "lab6.json"


def make(tmp_path):
    spec = {**load_spec(SPEC), "deadline": None}
    bench.make_clean_repo(tmp_path, spec)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return p


def test_cli_ok_exit_0_and_history(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "")
    sp = make(tmp_path)
    hist = tmp_path / "h.jsonl"
    assert main(["check", str(tmp_path), "--spec", str(sp), "--history", str(hist), "--md", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "ГОТОВО до здачі" in out and "AI-суддівство пропущено" in out  # без ключа — детерміновані перевірки все одно працюють
    assert json.loads(hist.read_text(encoding="utf-8"))["errors"] == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["errors"] == 0
    assert "# Звіт перевірки" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_cli_exit_1_on_error(tmp_path, capsys):
    sp = make(tmp_path)
    (tmp_path / "lab6" / "01_embeddings.ipynb").unlink()
    assert main(["check", str(tmp_path), "--spec", str(sp), "--no-ai"]) == 1
    assert "НЕ ГОТОВО" in capsys.readouterr().out


def test_cli_exit_2_without_spec_or_bad_root(tmp_path, capsys):
    assert main(["check", str(tmp_path), "--no-ai"]) == 2
    assert main(["check", str(tmp_path / "нема"), "--spec", str(SPEC), "--no-ai"]) == 2


def test_cli_assignment_without_key_falls_back_to_spec(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    sp = make(tmp_path)
    inp = Path(__file__).resolve().parents[1] / "inputs" / "lab6_assignment.txt"
    assert main(["check", str(tmp_path), "--spec", str(sp), "--assignment", str(inp)]) == 0
    assert "AI-розбір завдання пропущено" in capsys.readouterr().out

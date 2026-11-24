"""Читання ноутбуків .ipynb і markdown-звітів у єдиному вигляді «клітинок» + пошук блоків відповідей на питання."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class NotebookError(Exception):
    pass


@dataclass
class Cell:
    idx: int
    kind: str  # markdown | code
    source: str
    exec_count: int | None = None
    out_text: str = ""
    n_outputs: int = 0
    has_error_output: bool = False
    error_name: str = ""


@dataclass
class Notebook:
    path: Path
    cells: list[Cell]
    metadata: dict = field(default_factory=dict)
    is_markdown_file: bool = False


def read_text_any(path: Path) -> str:
    """Читає файл у UTF-8 (з BOM чи без), інакше пробує cp1251 (типово для Windows-файлів)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _join(x) -> str:
    return "".join(x) if isinstance(x, list) else (x or "")


def _output_text(outputs: list) -> tuple[str, bool, str]:
    parts: list[str] = []
    err, ename = False, ""
    for o in outputs or []:
        t = o.get("output_type")
        if t == "stream":
            parts.append(_join(o.get("text")))
        elif t in ("execute_result", "display_data"):
            parts.append(_join((o.get("data") or {}).get("text/plain")))
        elif t == "error":
            err, ename = True, str(o.get("ename", "Error"))
            parts.append(f"{ename}: {o.get('evalue', '')}")
    return "\n".join(p for p in parts if p), err, ename


def load_notebook(path: Path) -> Notebook:
    try:
        data = json.loads(read_text_any(path))
    except json.JSONDecodeError as e:
        raise NotebookError(f"не JSON: {e.msg} (рядок {e.lineno})") from e
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise NotebookError("немає списку cells (це не nbformat 4)")
    cells: list[Cell] = []
    for i, c in enumerate(data["cells"]):
        if not isinstance(c, dict):
            raise NotebookError(f"клітинка {i} має неправильний формат")
        kind = c.get("cell_type", "")
        text, err, ename = _output_text(c.get("outputs", [])) if kind == "code" else ("", False, "")
        cells.append(Cell(i, kind, _join(c.get("source")), c.get("execution_count"), text,
                          len(c.get("outputs", []) or []) if kind == "code" else 0, err, ename))
    return Notebook(path, cells, data.get("metadata") or {})


_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S")


def load_markdown_file(path: Path) -> Notebook:
    """Markdown-файл (report.md) розрізаємо на «клітинки» за заголовками, щоб працювала та сама логіка пошуку."""
    text = read_text_any(path)
    cells: list[Cell] = []
    buf: list[str] = []
    in_code = False

    def flush():
        if buf and "".join(buf).strip():
            cells.append(Cell(len(cells), "markdown", "\n".join(buf).strip("\n")))
        buf.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and _HEADING.match(line):
            flush()
        buf.append(line)
    flush()
    return Notebook(path, cells, {}, is_markdown_file=True)


def load_any(path: Path) -> Notebook:
    return load_markdown_file(path) if path.suffix.lower() == ".md" else load_notebook(path)


# ---------- пошук питань ----------
_KW = r"(?:питання|запитання|question|відповідь|q)"
_Q_KEYWORD = re.compile(rf"^\W*{_KW}\s*[№#]?\s*(\d{{1,3}})\b", re.I)
_Q_BARE = re.compile(r"^\s{0,3}#{3,6}\s*\**\s*(\d{1,3})[\.\)]\s")  # лише ### і глибше: «## 1. Розділ» не питання


def heading_level(first_line: str) -> int:
    m = _HEADING.match(first_line)
    return len(m.group(1)) if m else 0


def question_number(cell: Cell, md_file: bool = False) -> int | None:
    """Номер питання, якщо клітинка — заголовок відповіді («### Питання 7. ...», «### 7. ...»)."""
    if cell.kind != "markdown":
        return None
    first = next((l for l in cell.source.splitlines() if l.strip()), "")
    m = _Q_KEYWORD.match(first.lstrip("# *_").strip()) if first.lstrip().startswith(("#", "*", "_")) else None
    if m is None and first.strip():
        stripped = re.sub(r"^[#>\s*_]+", "", first)
        m = _Q_KEYWORD.match(stripped) if first.lstrip().startswith(("#", "**")) else None
    if m:
        return int(m.group(1))
    if not md_file:  # у ноутбуках «### 1. Розділ демо» — звичайні розділи, а не питання
        return None
    m2 = _Q_BARE.match(first)
    return int(m2.group(1)) if m2 else None


@dataclass
class Block:
    n: int
    heading_idx: int
    heading: str
    cells: list[Cell]
    answer_words: int
    code_cells: int

    def text(self, limit: int = 3500) -> str:
        """Текст блоку для суддівства: заголовок, markdown, початок коду і текстових виводів."""
        parts: list[str] = []
        for c in self.cells:
            if c.kind == "markdown":
                parts.append(c.source.strip())
            else:
                src = c.source.strip()
                parts.append("[код]\n" + (src[:700] + (" …" if len(src) > 700 else "")))
                if c.out_text.strip():
                    o = c.out_text.strip()
                    parts.append("[вивід]\n" + (o[:500] + (" …" if len(o) > 500 else "")))
        full = "\n\n".join(parts)
        if len(full) > limit:
            half = limit // 2
            full = full[:half] + "\n[…пропущено…]\n" + full[-half:]
        return full


_WORD = re.compile(r"[\w'’ʼ-]+", re.U)


def find_blocks(nb: Notebook) -> dict[int, Block]:
    """Усі блоки відповідей у ноутбуці: {номер_у_файлі: Block}. Дубль номера — береться перший."""
    heads = [(c.idx, question_number(c, nb.is_markdown_file)) for c in nb.cells]
    heads = [(i, n) for i, n in heads if n is not None]
    out: dict[int, Block] = {}
    for pos, (i, n) in enumerate(heads):
        if n in out:
            continue
        hc = nb.cells[i]
        first = next((l for l in hc.source.splitlines() if l.strip()), "")
        level = heading_level(first) or 3
        next_q = heads[pos + 1][0] if pos + 1 < len(heads) else len(nb.cells)
        end = next_q
        for j in range(i + 1, next_q):
            cj = nb.cells[j]
            if cj.kind == "markdown":
                fl = next((l for l in cj.source.splitlines() if l.strip()), "")
                lv = heading_level(fl)
                if lv and lv <= level:
                    end = j
                    break
        cells = nb.cells[i:end]
        body_words = 0
        for k, c in enumerate(cells):
            if c.kind != "markdown":
                continue
            src = c.source
            if k == 0:  # у клітинці-заголовку відкидаємо сам заголовок
                lines = src.splitlines()
                idx0 = next((x for x, l in enumerate(lines) if l.strip()), 0)
                src = "\n".join(lines[idx0 + 1:])
            body_words += len(_WORD.findall(src))
        out[n] = Block(n, i, first.strip(), cells, body_words, sum(1 for c in cells if c.kind == "code"))
    return out


def markdown_lines(nb: Notebook) -> list[str]:
    lines: list[str] = []
    for c in nb.cells:
        if c.kind == "markdown":
            lines.extend(l.strip() for l in c.source.splitlines() if l.strip())
    return lines

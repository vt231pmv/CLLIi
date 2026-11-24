"""Детерміновані перевірки: працюють без мережі й без AI, дають повторюваний результат.

Кожна перевірка повертає Finding(sev, code, where, msg). Коди:
  E_* — помилка (здавати не можна), W_* — попередження, I_* — інформація.
"""
from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import notebook as nbm
from .spec import top_dirs

ERR, WARN, INFO = "ERR", "WARN", "INFO"

PLACEHOLDER_PATTERNS = [
    r"ПІСЛЯ\s+ЗАПУСКУ", r"⏳", r"\bTODO\b", r"\bFIXME\b", r"\bTBD\b", r"\bXXX\b",
    r"(?i)ваша\s+відповідь", r"(?i)\byour\s+answer\s+here\b", r"(?i)тут\s+(?:має|буде)\s+(?:бути\s+)?відповідь",
    r"(?i)<[^<>\n]*(?:відповідь|answer|your)[^<>\n]*>",
    r"(?m)^\s*(?:\.\.\.|…)\s*$",          # рядок із самих крапок
    r"(?m):\s*(?:\.\.\.|…)\s*$",          # «Корпус і чому потрібен RAG: ...» із шаблону завдання
]
SECRET_PATTERNS = {
    "OpenAI/Anthropic-подібний ключ": r"\bsk-[A-Za-z0-9_\-]{20,}",
    "Google API key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "GitHub token": r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}",
    "HuggingFace token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "приватний ключ": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "присвоєння секрету": r"(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b\s*[=:]\s*['\"][A-Za-z0-9_\-./+]{16,}['\"]",
}
FORBIDDEN_EXT = {
    "model_weights": {".safetensors", ".gguf", ".ggml", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".tflite"},
    "vector_index": {".faiss", ".annoy", ".hnsw", ".index"},
}
FORBIDDEN_DIRS = {
    "vector_index": {"faiss_index", "chroma", "chroma_db", "vectorstore", "vector_store", "qdrant_storage", "index_store"},
    "model_weights": {"adapter", "lora_adapter", "checkpoints"},
}
JUNK_NAMES = {".ipynb_checkpoints", "__pycache__", ".DS_Store", "Thumbs.db", ".idea", ".vscode"}
SKIP_WALK = {".git", "node_modules", ".venv", "venv", "env"}
BIG_WARN, BIG_ERR = 50 * 2**20, 100 * 2**20  # GitHub: попередження від 50 МБ, відхилення від 100 МБ
BIN_WEIGHTS_MIN = 5 * 2**20  # *.bin більше 5 МБ вважаємо вагами


@dataclass
class Finding:
    sev: str
    code: str
    where: str
    msg: str

    def as_dict(self) -> dict:
        return {"sev": self.sev, "code": self.code, "where": self.where, "msg": self.msg}


@dataclass
class QuestionStatus:
    n: int
    file: str
    text: str
    found: bool = False
    heading: str = ""
    words: int = 0
    code_cells: int = 0
    block_text: str = ""
    # заповнюється AI-суддею
    verdict: str = ""
    reason: str = ""
    quote: str = ""
    rejected_quote: str = ""  # цитата моделі, якої не знайдено в тексті (для діагностики)


@dataclass
class StaticResult:
    base: Path
    findings: list[Finding] = field(default_factory=list)
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, статус)
    questions: list[QuestionStatus] = field(default_factory=list)
    notebooks_checked: int = 0

    def add(self, sev, code, where, msg):
        self.findings.append(Finding(sev, code, where, msg))

    def count(self, sev: str) -> int:
        return sum(1 for f in self.findings if f.sev == sev)



# ---------- .gitignore (спрощено): що git не віддасть у репозиторій, те й не вважаємо проблемою ----------
class GitIgnore:
    def __init__(self, base: Path):
        self.base = base
        self._cache: dict[Path, list[str]] = {}

    def _patterns(self, d: Path) -> list[str]:
        if d not in self._cache:
            pats: list[str] = []
            gi = d / ".gitignore"
            if gi.is_file():
                for line in nbm.read_text_any(gi).splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", "!")):
                        pats.append(line)
            self._cache[d] = pats
        return self._cache[d]

    def ignored(self, path: Path) -> bool:
        try:
            rel_parts = path.relative_to(self.base).parts
        except ValueError:
            return False
        import fnmatch
        for depth in range(len(rel_parts)):  # .gitignore у кожному каталозі на шляху від бази до елемента
            d = self.base.joinpath(*rel_parts[:depth])
            sub = rel_parts[depth:]
            for pat in self._patterns(d):
                pat = pat.rstrip("/")
                anchored = pat.startswith("/")
                pat = pat.lstrip("/")
                if anchored or "/" in pat:
                    if fnmatch.fnmatch("/".join(sub), pat) or any(fnmatch.fnmatch("/".join(sub[:k]), pat) for k in range(1, len(sub) + 1)):
                        return True
                elif any(fnmatch.fnmatch(part, pat) for part in sub):
                    return True
        return False


# ---------- шляхи ----------
def resolve_base(root: Path, spec: dict) -> Path:
    """Якщо користувач запустив усередині lab6/ або mkr1/, базою вважаємо батьківську теку."""
    root = root.resolve()  # «..» або «.» не мають імені, а нам потрібно знати, чи тека називається lab6/mkr1
    tops = top_dirs(spec)
    if not tops:
        return root
    if any((root / t).is_dir() for t in tops):
        return root
    if root.name in tops:
        return root.parent
    return root


def exact_exists(base: Path, rel: str, want_dir: bool = False) -> bool:
    """Існування з урахуванням РЕГІСТРУ (GitHub чутливий до регістру, Windows — ні)."""
    cur = base
    for part in Path(rel).parts:
        try:
            names = {p.name for p in cur.iterdir()}
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return False
        if part not in names:
            return False
        cur = cur / part
    return cur.is_dir() if want_dir else cur.is_file()


def _walk(scope: Path):
    for dirpath, dirnames, filenames in __import__("os").walk(scope):
        dirnames[:] = [d for d in dirnames if d not in SKIP_WALK]
        yield Path(dirpath), dirnames, filenames


# ---------- окремі перевірки ----------
def check_required(res: StaticResult, spec: dict, base: Path) -> None:
    for r in spec["required"]:
        rel, kind = r["path"], r.get("type", "file")
        want_dir = kind == "dir"
        if exact_exists(base, rel, want_dir):
            if want_dir and not any((base / rel).iterdir()):
                res.files.append((rel, "порожня тека"))
                res.add(ERR, "E_EMPTY_DIR", rel, "тека є, але порожня")
            else:
                res.files.append((rel, "є"))
            continue
        res.files.append((rel, "НЕМАЄ"))
        hint = _near_miss(base, rel, spec)
        res.add(ERR, "E_MISSING", rel, "обов'язкового " + ("каталогу" if want_dir else "файлу") + " немає за точним шляхом" + (f". Схоже: {hint}" if hint else ""))


def _near_miss(base: Path, rel: str, spec: dict) -> str:
    name = Path(rel).name
    best: list[tuple[float, str]] = []
    scopes = [base] + [base / t for t in top_dirs(spec) if (base / t).is_dir()]
    seen = set()
    for sc in scopes:
        if not sc.is_dir():
            continue
        for p in sc.rglob("*") if sc == base and False else sc.iterdir():
            if p in seen:
                continue
            seen.add(p)
            ratio = difflib.SequenceMatcher(None, name.lower(), p.name.lower()).ratio()
            if p.name != name and ratio >= 0.72:
                best.append((ratio, str(p.relative_to(base)).replace("\\", "/")))
    best.sort(reverse=True)
    return ", ".join(x for _, x in best[:2])


def check_tree(res: StaticResult, spec: dict, base: Path) -> None:
    forb = {f["kind"] for f in spec.get("forbidden", [])}
    gi = GitIgnore(base)

    def add(path: Path, sev, code, where, msg):
        if gi.ignored(path):  # такого файлу в репозиторії не буде: це не помилка, лише примітка
            res.add(INFO, "I_GITIGNORED", where, f"{code} пропущено: шлях у .gitignore ({msg[:60]})")
        else:
            res.add(sev, code, where, msg)

    for top in top_dirs(spec) or ["."]:
        scope = base / top
        if not scope.is_dir():
            continue
        for dirpath, dirnames, filenames in _walk(scope):
            rel_dir = dirpath.relative_to(base).as_posix()
            for d in list(dirnames):
                dp = dirpath / d
                if d in JUNK_NAMES:
                    add(dp, WARN, "W_JUNK", f"{rel_dir}/{d}", "службова тека не має потрапляти в репозиторій")
                for kind in forb:
                    if d.lower() in FORBIDDEN_DIRS.get(kind, ()):
                        add(dp, ERR, "E_FORBIDDEN", f"{rel_dir}/{d}", f"тека схожа на заборонене за завданням ({kind})")
                if d in (".venv", "venv"):
                    add(dp, WARN, "W_JUNK", f"{rel_dir}/{d}", "віртуальне середовище не додають до репозиторію")
            for fn in filenames:
                p = dirpath / fn
                rel = p.relative_to(base).as_posix()
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                ext = p.suffix.lower()
                if fn in JUNK_NAMES:
                    add(p, WARN, "W_JUNK", rel, "службовий файл")
                if fn == ".env" or (fn.startswith(".env.") and fn not in (".env.example", ".env.sample", ".env.template")):
                    add(p, ERR, "E_SECRET", rel, "файл змінних середовища майже напевно містить ключі")
                for kind in forb:
                    if ext in FORBIDDEN_EXT.get(kind, ()) or (kind == "model_weights" and ext == ".bin" and size > BIN_WEIGHTS_MIN):
                        add(p, ERR, "E_FORBIDDEN", rel, f"заборонено завданням ({kind}), розмір {_mb(size)}")
                    elif kind == "vector_index" and ext in (".pkl", ".npy", ".parquet") and size > BIN_WEIGHTS_MIN and re.search(r"index|vector|embed", fn, re.I):
                        add(p, ERR, "E_FORBIDDEN", rel, f"схоже на векторний індекс/ембединги ({_mb(size)})")
                if size >= BIG_ERR:
                    add(p, ERR, "E_TOO_BIG", rel, f"{_mb(size)} ≥ 100 МБ: GitHub відхилить пуш")
                elif size >= BIG_WARN:
                    add(p, WARN, "W_BIG", rel, f"{_mb(size)}: GitHub попередить про великий файл")
                if ext in (".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".log", ".sh", ".bat") and size < 2 * 2**20 and not gi.ignored(p):
                    _scan_secrets(res, rel, nbm.read_text_any(p))


def _mb(n: int) -> str:
    return f"{n / 2**20:.1f} МБ"


def _scan_secrets(res: StaticResult, where: str, text: str) -> None:
    for label, pat in SECRET_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            s = m.group(0)
            res.add(ERR, "E_SECRET", where, f"{label}: {s[:4]}…{'*' * 6} (значення приховано)")
            return


def check_placeholders(res: StaticResult, rel: str, nb: nbm.Notebook, patterns=None) -> None:
    pats = [re.compile(p) for p in (patterns or PLACEHOLDER_PATTERNS)]
    hits: list[int] = []
    for c in nb.cells:
        if c.kind != "markdown":
            continue
        if any(p.search(c.source) for p in pats):
            hits.append(c.idx)
    if hits:
        shown = ", ".join(str(i) for i in hits[:8]) + (" …" if len(hits) > 8 else "")
        res.add(ERR, "E_PLACEHOLDER", rel, f"залишилось плейсхолдерів у markdown-клітинках: {len(hits)} (клітинки {shown})")


def check_notebook_quality(res: StaticResult, rel: str, nb: nbm.Notebook) -> None:
    if nb.is_markdown_file:
        return
    errs = [c for c in nb.cells if c.has_error_output]
    if errs:
        names = sorted({c.error_name for c in errs})
        res.add(ERR, "E_CELL_ERROR", rel, f"у виводах є помилки виконання ({len(errs)} клітинок: {', '.join(names[:3])})")
    code = [c for c in nb.cells if c.kind == "code" and _meaningful(c.source)]
    notrun = [c for c in code if c.exec_count is None and not c.n_outputs]
    if code and notrun:
        share = len(notrun) / len(code)
        sev = ERR if share > 0.5 else WARN
        res.add(sev, "W_NOT_EXECUTED", rel, f"клітинок коду без запуску: {len(notrun)} із {len(code)} (завдання вимагає виконаних клітинок)")
    wid = nb.metadata.get("widgets")
    if isinstance(wid, dict):
        bad = [k for k, v in wid.items() if isinstance(v, dict) and "state" not in v]
        if bad:
            res.add(WARN, "W_WIDGETS_STATE", rel, "metadata.widgets без ключа state: GitHub не відрендерить ноутбук («Invalid Notebook»)")
    secrets_text = "\n".join(c.source + "\n" + c.out_text for c in nb.cells)
    _scan_secrets(res, rel, secrets_text)


def _meaningful(src: str) -> bool:
    for line in src.splitlines():
        s = line.strip()
        if s and not s.startswith(("#", "%", "!")):
            return True
    return False


def _tokens(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w+", s.lower()) if len(w) > 2}


def check_sections(res: StaticResult, spec: dict, loaded: dict[str, nbm.Notebook]) -> None:
    """Назва секції має бути в заголовку, у жирному рядку або в короткому рядку. Для 1–2-слівних назв («Висновок»)
    дозволені лише заголовки/жирне: інакше слово з будь-якого абзацу «знайде» секцію, якої немає."""
    by_file: dict[str, list[dict]] = {}
    for s in spec.get("sections", []):
        by_file.setdefault(s.get("file") or "", []).append(s)
    for rel, secs in by_file.items():
        nb = loaded.get(rel)
        if nb is None:
            continue
        cands: list[tuple[bool, set[str]]] = []
        for raw in nbm.markdown_lines(nb):
            head = raw.startswith(("#", "**", "__")) or len(raw.split()) <= 12
            strong = raw.startswith(("#", "**", "__"))
            cands.append((strong, _tokens(raw.lstrip("#*_ >-").strip())) if head else (False, set()))
        for s in secs:
            need = _tokens(s["title"])
            if not need:
                continue
            short = len(need) <= 2
            ok = any((strong or not short) and tk and len(need & tk) / len(need) >= 0.6 for strong, tk in cands)
            if not ok:
                res.add(WARN, "W_SECTION", rel, f"не знайдено секції звіту «{s['title']}»")


def check_media(res: StaticResult, spec: dict, base: Path) -> None:
    for m in spec.get("media", []):
        p = base / m["path"]
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size < 50_000:
            res.add(WARN, "W_MEDIA", m["path"], f"файл підозріло малий ({size} Б)")
        dur = _probe_duration(p)
        if dur is None:
            res.add(INFO, "I_MEDIA", m["path"], "тривалість не перевірено (немає ffprobe); вимога: "
                    f"{m.get('min_s', 0)}–{m.get('max_s', '∞')} с")
        elif not (m.get("min_s", 0) <= dur <= m.get("max_s", 10**9)):
            res.add(ERR, "E_MEDIA_LEN", m["path"], f"тривалість {dur:.0f} с, потрібно {m.get('min_s', 0)}–{m.get('max_s', '∞')} с")
        else:
            res.add(INFO, "I_MEDIA", m["path"], f"тривалість {dur:.0f} с — у межах вимоги")


def _probe_duration(p: Path) -> float | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                             capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def check_questions(res: StaticResult, spec: dict, loaded: dict[str, nbm.Notebook], thin_words: int = 15) -> None:
    by_file: dict[str, list[dict]] = {}
    for q in spec.get("questions", []):
        by_file.setdefault(q.get("file") or "", []).append(q)
    for rel, qs in by_file.items():
        nb = loaded.get(rel)
        blocks = nbm.find_blocks(nb) if nb else {}
        want = sorted(q["n"] for q in qs)
        shift = 0
        if blocks and not set(want) <= set(blocks) and set(blocks) == set(range(1, len(want) + 1)):
            shift = want[0] - 1  # локальна нумерація 1..k замість глобальної
            res.add(INFO, "I_LOCAL_NUMBERING", rel, f"у файлі нумерація 1–{len(want)} замість {want[0]}–{want[-1]}; зіставляю за порядком")
        for q in sorted(qs, key=lambda x: x["n"]):
            st = QuestionStatus(q["n"], rel, q["text"])
            b = blocks.get(q["n"] - shift)
            if b is None:
                where = rel or "(файл не вказано)"
                reason = "файл не завантажено" if nb is None else "заголовок відповіді не знайдено"
                res.add(ERR if nb is not None else WARN, "E_Q_MISSING" if nb is not None else "W_Q_SKIPPED", where, f"питання {q['n']}: {reason}")
            else:
                st.found, st.heading, st.words, st.code_cells = True, b.heading, b.answer_words, b.code_cells
                st.block_text = b.text()
                if b.answer_words < thin_words:  # навіть якщо поруч є код: відповідь має бути словами
                    res.add(WARN, "W_Q_THIN", rel, f"питання {q['n']}: текстова відповідь надто коротка ({b.answer_words} слів, клітинок коду: {b.code_cells})")
            res.questions.append(st)


def check_deadline(res: StaticResult, spec: dict, today: date | None = None) -> None:
    d = spec.get("deadline")
    if not d:
        return
    today = today or date.today()
    try:
        left = (date.fromisoformat(d) - today).days
    except ValueError:
        return
    if left < 0:
        res.add(ERR, "E_DEADLINE", d, f"дедлайн минув {-left} дн. тому")
    else:
        res.add(INFO, "I_DEADLINE", d, f"до дедлайну {left} дн.")


def run_static(root: Path, spec: dict, today: date | None = None) -> StaticResult:
    base = resolve_base(root, spec)
    res = StaticResult(base)
    check_required(res, spec, base)
    check_tree(res, spec, base)
    loaded: dict[str, nbm.Notebook] = {}
    wanted = {r["path"] for r in spec["required"] if r.get("type", "file") == "file"}
    wanted |= {q["file"] for q in spec.get("questions", []) if q.get("file")}
    wanted |= {s["file"] for s in spec.get("sections", []) if s.get("file")}
    for rel in sorted(wanted):
        p = base / rel
        if not p.is_file() or not exact_exists(base, rel):
            continue
        if p.suffix.lower() not in (".ipynb", ".md"):
            continue
        try:
            nb = nbm.load_any(p)
        except nbm.NotebookError as e:
            res.add(ERR, "E_BAD_JSON", rel, f"ноутбук не відкривається: {e}")
            continue
        except OSError as e:
            res.add(ERR, "E_IO", rel, f"не вдалося прочитати: {e}")
            continue
        loaded[rel] = nb
        res.notebooks_checked += 1
        check_placeholders(res, rel, nb)
        check_notebook_quality(res, rel, nb)
    check_questions(res, spec, loaded)
    check_sections(res, spec, loaded)
    check_media(res, spec, base)
    check_deadline(res, spec, today)
    return res

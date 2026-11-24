"""Два AI-кроки рівня 1 (один запит до моделі кожен) + код, що перевіряє відповіді моделі.

Принцип: AI пропонує, код перевіряє. Усе, що модель «процитувала», звіряється з першоджерелом;
не підтверджене — не приймається. AI-знахідки завжди лише попередження, помилки дає тільки детермінований код.
"""
from __future__ import annotations

import difflib
import json
import re
from datetime import date

from .llm import LLMClient, LLMUnavailable, parse_json_loose
from .spec import FORBIDDEN_KINDS, validate_spec
from .static_checks import QuestionStatus

# ---------------------------------------------------------------- нормалізація
_MD = re.compile(r"[*_`#>|]+")


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("ʼ", "'").replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", _MD.sub(" ", s)).strip().lower()


_TOK = re.compile(r"\w+")
_ELLIPSIS = re.compile(r"\[?\s*(?:\.{3,}|…)\s*\]?")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(norm(s))


def _find(hay: list[str], needle: list[str], start: int = 0) -> int:
    return next((k for k in range(start, len(hay) - len(needle) + 1) if hay[k:k + len(needle)] == needle), -1)


def quote_in(quote: str, text: str) -> bool:
    """Чи є цитата моделі в тексті ДОСЛІВНО за словами. Порівнюємо послідовності слів: регістр, пунктуація, розмітка
    markdown і пробіли навколо дужок/лапок значення не мають (модель їх часто «чистить»); пропущене слово чи переказ — мають.
    Якщо цілою цитати немає, а в ній є «…» / «...», цитата вважається набором фрагментів: кожен має бути в тексті, у тому ж порядку."""
    hay = _tokens(text)
    whole = _tokens(quote)
    if len(whole) < 3 or len("".join(whole)) < 6:   # одне-два слова нічого не доводять
        return False
    if _find(hay, whole) >= 0:
        return True
    frags = [f for f in (_tokens(x) for x in _ELLIPSIS.split(str(quote)) if x.strip()) if f]
    if len(frags) < 2 or any(len(f) < 2 or len("".join(f)) < 6 for f in frags):
        return False
    pos = 0
    for f in frags:
        i = _find(hay, f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


# ---------------------------------------------------------------- шляхи
_PATH_LINE = re.compile(r"^\s*(?:[-*•]\s*)?/([\w.\-]+(?:/[\w.\-*]*)+)", re.M)


def norm_path(p) -> str:
    """Єдина форма шляху: без початкового «/» чи «./», без кінцевих «/», «/*», «*»; зворотні слеші → прямі."""
    p = str(p).strip().strip("`'\"").replace("\\", "/")
    p = re.sub(r"^(?:\./|/)+", "", p)
    return re.sub(r"[/*]+$", "", p)


def canonical_paths(text: str) -> dict[str, str]:
    """Повні шляхи, записані в тексті окремими рядками виду «/mkr1/report.md» або «/mkr1/solution/*» → {шлях: file|dir}.
    Це детермінований розбір рядків; модель його лише НЕ повинна суперечити."""
    out: dict[str, str] = {}
    for m in _PATH_LINE.finditer(text.replace("\\", "/")):
        raw = m.group(1)
        path = norm_path(raw)
        if "/" not in path:
            continue
        is_dir = raw.endswith(("/", "*")) or not re.search(r"\.\w{1,6}$", path)
        out.setdefault(path, "dir" if is_dir else "file")
    return out


def resolve_path(p, known) -> str | None:
    """Зіставляє шлях моделі з відомими повними шляхами: точний збіг, або єдиний, що закінчується на «/<p>»
    (модель написала `report.md`, а в завданні `/mkr1/report.md`). Не знайшов або неоднозначно → None."""
    p = norm_path(p)
    if not p:
        return None
    if p in known:
        return p
    cands = [k for k in known if k.endswith("/" + p)]
    return cands[0] if len(cands) == 1 else None


# ---------------------------------------------------------------- крок A: розбір завдання
EXTRACT_SYSTEM = (
    "Ти розбираєш текст навчального завдання на чекліст для самоперевірки перед здачею. "
    "Відповідай ЛИШЕ одним валідним JSON-об'єктом, без пояснень і без markdown-огорож. "
    "Нічого не вигадуй: усе, що кладеш у JSON, має прямо випливати з тексту завдання."
)
EXTRACT_USER = """Витягни з тексту завдання:
- title: назва роботи;
- deadline: дата здачі у форматі YYYY-MM-DD або null;
- required: обов'язкові файли/каталоги репозиторію з розділу про структуру здачі, де шляхи записані повністю від кореня репозиторію (рядки на кшталт /work3/notes.md). Пиши ПОВНИЙ шлях від кореня разом з першою папкою (work3/notes.md), без початкового «/», без «*» і без кінцевого «/»; один запис на кожен рядок структури; згадки файлів в інших місцях тексту окремими записами НЕ додавай. type: "file" або "dir" (dir — якщо запис закінчується на «/*» чи «/»); quote: дослівний уривок (до 12 слів) із тексту;
- forbidden: що заборонено додавати до репозиторію; kind лише "vector_index" або "model_weights"; quote — дослівно;
- media: вимоги до відео (path, min_s, max_s, quote) або [];
- questions: ВСІ теоретичні питання зі списку, нумеруй за порядком у списку від 1; text — дослівно як у завданні; file — в який з обов'язкових файлів за правилом розподілу в розділі про здачу потрапляє відповідь (повний шлях рівно в тій самій формі, що в required) або null;
- sections: назви обов'язкових розділів власного звіту (з шаблону «Мій звіт»), без двокрапки й крапок; file — файл, де вони мають бути, або null.

Формат:
{"title":"","deadline":null,"required":[{"path":"","type":"file","quote":""}],"forbidden":[{"kind":"","quote":""}],"media":[],"questions":[{"n":1,"text":"","file":""}],"sections":[{"title":"","file":""}]}

ТЕКСТ ЗАВДАННЯ:
<<<
{text}
>>>"""

REPAIR_USER = "Твоя попередня відповідь не є валідним JSON-об'єктом. Поверни ТІЛЬКИ JSON за тим самим форматом, без жодного іншого тексту."


def _ask_json(llm: LLMClient, system: str, user: str, max_tokens: int) -> dict:
    r = llm.chat(system, user, max_tokens)
    try:
        d = parse_json_loose(r.text)
    except ValueError:
        r2 = llm.chat(system, user + "\n\n" + REPAIR_USER, max_tokens)
        d = parse_json_loose(r2.text)  # другий провал → ValueError нагору
    if not isinstance(d, dict):
        raise ValueError("очікувався JSON-об'єкт")
    return d


def extract_spec(llm: LLMClient, assignment_text: str) -> tuple[dict, dict]:
    """AI розбирає завдання → код перевіряє кожне твердження за текстом. Повертає (spec, статистика перевірки)."""
    raw = _ask_json(llm, EXTRACT_SYSTEM, EXTRACT_USER.replace("{text}", assignment_text), 12000)
    return verify_spec(raw, assignment_text)


def verify_spec(raw: dict, text: str) -> tuple[dict, dict]:
    nt = norm(text)
    flat = text.replace("\\", "/")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    nlines = [norm(l) for l in lines]
    st = {k: 0 for k in ("required_kept", "required_dropped", "forbidden_dropped", "questions_kept",
                         "questions_healed", "questions_dropped", "sections_dropped", "files_nulled", "paths_normalized",
                         "required_duplicates")}

    canon = canonical_paths(text)
    required: list[dict] = []
    for r in raw.get("required") or []:
        raw_path = str(r.get("path", "")) if isinstance(r, dict) else ""
        path = resolve_path(raw_path, canon) or norm_path(raw_path)
        if path and (path in canon or path in flat) and path not in {x["path"] for x in required}:
            typ = canon.get(path) or (r.get("type") if r.get("type") in ("file", "dir") else ("file" if re.search(r"\.\w{1,6}$", path) else "dir"))
            required.append({"path": path, "type": typ, "quote": str(r.get("quote", ""))})
            st["required_kept"] += 1
            if path != raw_path.strip():
                st["paths_normalized"] += 1  # код привів шлях до повної форми (прибрав «/*», додав першу папку тощо)
        elif path in {x["path"] for x in required}:
            st["required_duplicates"] += 1
        else:
            st["required_dropped"] += 1  # модель вигадала шлях, якого немає в тексті
    paths = {r["path"] for r in required}

    forbidden = []
    for f in raw.get("forbidden") or []:
        if isinstance(f, dict) and f.get("kind") in FORBIDDEN_KINDS and quote_in(str(f.get("quote", "")), text):
            forbidden.append({"kind": f["kind"], "quote": f["quote"]})
        else:
            st["forbidden_dropped"] += 1

    media = []
    for m in raw.get("media") or []:
        try:
            mp = resolve_path(m["path"], paths) if isinstance(m, dict) else None
            if mp and quote_in(str(m.get("quote", "")), text):
                media.append({"path": mp, "min_s": int(m["min_s"]), "max_s": int(m["max_s"]), "quote": m["quote"]})
        except (KeyError, ValueError, TypeError):
            pass

    questions, seen = [], set()
    for q in raw.get("questions") or []:
        if not (isinstance(q, dict) and isinstance(q.get("n"), int) and isinstance(q.get("text"), str)) or q["n"] in seen:
            st["questions_dropped"] += 1
            continue
        nq = norm(q["text"])
        core = nq.rstrip(" ?.!:;")
        idx = next((i for i, nl in enumerate(nlines) if len(core) >= 12 and core[:60] in nl), None)
        if idx is None:
            close = difflib.get_close_matches(nq, nlines, n=1, cutoff=0.85)
            idx = nlines.index(close[0]) if close else None
        if idx is None:
            st["questions_dropped"] += 1  # питання немає в тексті завдання
            continue
        qt = lines[idx]  # завжди беремо оригінальний рядок із завдання, а не переказ моделі
        if nlines[idx] != nq:
            st["questions_healed"] += 1
        f = q.get("file")
        if f is not None:
            f = resolve_path(f, paths)
            if f is None:
                st["files_nulled"] += 1
        questions.append({"n": q["n"], "text": qt, "file": f})
        seen.add(q["n"])
        st["questions_kept"] += 1
    questions.sort(key=lambda x: x["n"])

    sections = []
    for s in raw.get("sections") or []:
        title = str(s.get("title", "")).strip(" :.") if isinstance(s, dict) else ""
        words = set(re.findall(r"\w+", norm(title)))
        ok = bool(words) and any(len(words & set(re.findall(r"\w+", nl))) / len(words) >= 0.8 for nl in nlines)
        if ok:
            f = resolve_path(s.get("file"), paths) if s.get("file") else None
            sections.append({"title": title, "file": f})
        else:
            st["sections_dropped"] += 1

    deadline = raw.get("deadline")
    if deadline:
        try:
            d = date.fromisoformat(str(deadline))
            if f"{d.day:02d}.{d.month:02d}.{d.year}" not in text:
                deadline = None
        except ValueError:
            deadline = None

    spec = {"title": str(raw.get("title") or "завдання"), "deadline": deadline, "required": required,
            "forbidden": forbidden, "media": media, "questions": questions, "sections": sections}
    problems = validate_spec(spec)
    if problems:
        raise ValueError("після перевірки специфікація непридатна: " + "; ".join(problems[:3]))
    return spec, st


# ---------------------------------------------------------------- крок B: суддівство відповідей
VERDICTS = ("answered", "partial", "off_topic", "missing")
JUDGE_SYSTEM = (
    "Ти — уважний перевіряльник навчальних робіт. Для кожного питання визнач, чи в наведеному тексті є відповідь саме на це питання. "
    "Правильність фактів НЕ оцінюй — лише наявність відповіді по суті. Відповідай ЛИШЕ валідним JSON без markdown-огорож."
)
JUDGE_USER = """Вердикти:
- answered: є конкретна відповідь по суті; якщо питання вимагає запустити код чи показати числа — є код або вивід;
- partial: відповідь є, але неповна (не всі частини питання, немає обіцяного коду/виводу, лише загальні слова);
- off_topic: текст відповідає на інше питання або не стосується цього;
- missing: відповіді немає (повтор питання, порожньо, плейсхолдер).
Заголовок із текстом питання сам по собі НЕ є відповіддю.
Для answered і partial поле quote — ДОСЛІВНИЙ уривок (до 15 слів) із тексту відповіді, не із заголовка. Для інших — порожній рядок.
reason — до 20 слів українською.

Формат: {"results":[{"n":1,"verdict":"answered","quote":"...","reason":"..."}]}

{items}"""
ITEM = "=== ПИТАННЯ {n} ===\nПитання: {q}\nТекст відповіді:\n<<<\n{a}\n>>>\n"
MIN_WORDS_FOR_AI = 15


def _answer_part(block_text: str) -> str:
    """Текст блоку без першого рядка-заголовка (щоб модель і перевірка цитат не спиралися на питання)."""
    parts = block_text.split("\n", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def apply_verdicts(items: list[QuestionStatus], results: list[dict]) -> None:
    by_n = {r.get("n"): r for r in results if isinstance(r, dict)}
    for it in items:
        r = by_n.get(it.n)
        if not r or r.get("verdict") not in VERDICTS:
            it.verdict, it.reason = "needs_review", "модель не повернула коректний вердикт — перевірте вручну"
            continue
        v, quote = r["verdict"], str(r.get("quote", "") or "")
        it.reason = str(r.get("reason", ""))[:200]
        if v in ("answered", "partial"):
            if quote_in(quote, _answer_part(it.block_text)):
                it.verdict, it.quote = v, quote
            else:  # захист від вигадки: вердикт «відповідь є» без підтвердженої цитати не приймаємо
                it.verdict, it.rejected_quote = "needs_review", quote[:200]
                it.reason = "цитату моделі не знайдено в тексті відповіді — вердикт не прийнято; перевірте вручну"
        else:
            it.verdict = v


def judge_questions(llm: LLMClient, statuses: list[QuestionStatus], batch: int = 5, max_calls: int = 40) -> dict:
    """Заповнює verdict/reason/quote. Порожні/короткі відповіді рішає код без AI. Повертає лічильники."""
    cnt = {"skipped_not_found": 0, "skipped_thin": 0, "sent": 0, "calls": 0, "failed_batches": 0}
    todo: list[QuestionStatus] = []
    for s in statuses:
        if not s.found:
            s.verdict, s.reason = "missing", "заголовок відповіді не знайдено (AI не викликався)"
            cnt["skipped_not_found"] += 1
        elif s.words < MIN_WORDS_FOR_AI and s.code_cells == 0:
            s.verdict, s.reason = "missing", f"лише {s.words} слів і немає коду (AI не викликався)"
            cnt["skipped_thin"] += 1
        else:
            todo.append(s)
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        if cnt["calls"] >= max_calls:
            for s in part:
                s.verdict, s.reason = "needs_review", "ліміт викликів AI вичерпано"
            continue
        items = "\n".join(ITEM.format(n=s.n, q=s.text, a=_answer_part(s.block_text) or "(порожньо)") for s in part)
        try:
            d = _ask_json(llm, JUDGE_SYSTEM, JUDGE_USER.replace("{items}", items), 3000)
            cnt["calls"] += 1
            apply_verdicts(part, d.get("results") if isinstance(d.get("results"), list) else [])
        except (LLMUnavailable, ValueError) as e:
            cnt["failed_batches"] += 1
            for s in part:
                s.verdict, s.reason = "needs_review", f"AI недоступний або відповів некоректно ({type(e).__name__}) — перевірте вручну"
        cnt["sent"] += len(part)
    return cnt

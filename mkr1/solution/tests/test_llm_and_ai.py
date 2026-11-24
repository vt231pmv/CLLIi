import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from labcheck import ai_steps
from labcheck.llm import LLMAuthError, LLMClient, LLMUnavailable, LLMResult, LLMStats, parse_json_loose
from labcheck.static_checks import QuestionStatus


# ---------- справжній локальний HTTP-сервер, що імітує OpenAI-сумісне API ----------
class Server:
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.calls += 1
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                code, body = outer.script.pop(0) if outer.script else (200, {"choices": [{"message": {"content": "{}"}}], "usage": {}})
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                if code == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


OK = (200, {"choices": [{"message": {"content": '{"a": 1}'}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})


def client(url, tmp_path=None, **kw):
    return LLMClient(url, "test-key", "m", cache_dir=tmp_path, sleep=lambda s: None, price_in=1.0, price_out=2.0, **kw)


def test_retries_on_429_then_succeeds():
    s = Server([(429, {}), (503, {}), OK])
    try:
        c = client(s.url)
        r = c.chat("sys", "usr")
        assert r.text == '{"a": 1}' and s.calls == 3
        assert c.stats.cost_usd == pytest.approx((100 * 1 + 20 * 2) / 1e6)
    finally:
        s.close()


def test_auth_error_is_not_retried():
    s = Server([(401, {})])
    try:
        with pytest.raises(LLMAuthError):
            client(s.url).chat("s", "u")
        assert s.calls == 1
    finally:
        s.close()


def test_gives_up_after_retries_and_error_hides_key():
    s = Server([(500, {})] * 10)
    try:
        with pytest.raises(LLMUnavailable) as e:
            client(s.url, retries=2).chat("s", "u")
        assert s.calls == 3 and "test-key" not in str(e.value)
    finally:
        s.close()


def test_cache_prevents_second_request(tmp_path):
    s = Server([OK])
    try:
        c = client(s.url, tmp_path)
        c.chat("s", "u")
        r2 = c.chat("s", "u")
        assert s.calls == 1 and r2.cached and c.stats.cache_hits == 1
    finally:
        s.close()


def test_corrupt_cache_file_is_ignored(tmp_path):
    s = Server([OK])
    try:
        c = client(s.url, tmp_path)
        (tmp_path / f"{c._key('s', 'u', 4096)}.json").write_text("{oops", encoding="utf-8")
        assert c.chat("s", "u").text == '{"a": 1}'
    finally:
        s.close()


def test_unreachable_server_is_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        client("http://127.0.0.1:9", retries=1, timeout=1).chat("s", "u")


@pytest.mark.parametrize("raw", ['{"a":1}', '```json\n{"a":1}\n```', 'Ось відповідь: {"a":1} Дякую'])
def test_parse_json_loose(raw):
    assert parse_json_loose(raw) == {"a": 1}


def test_parse_json_loose_fails_cleanly():
    with pytest.raises(ValueError):
        parse_json_loose("просто текст")


# ---------- захист від вигадок моделі ----------
TEXT = """Лабораторна 9. Тест
Дедлайн: 01.01.2027

0.1. Теоретичні питання
Чому небо блакитне?
Скільки буде два плюс два і чому саме так виходить?
1. Власна робота
Структура файлів:
/lab9/a.ipynb
/lab9/b.ipynb
Ваги моделей у репозиторій не додавайте.
"""


def test_verify_spec_drops_invented_path_and_question():
    raw = {"title": "T", "deadline": "2027-01-01",
           "required": [{"path": "lab9/a.ipynb", "type": "file", "quote": "/lab9/a.ipynb"}, {"path": "lab9/zzz.ipynb", "type": "file", "quote": "x"}],
           "forbidden": [{"kind": "model_weights", "quote": "Ваги моделей у репозиторій не додавайте"}, {"kind": "vector_index", "quote": "вигадана цитата якої немає"}],
           "questions": [{"n": 1, "text": "Чому небо блакитне?", "file": "lab9/a.ipynb"},
                         {"n": 2, "text": "Скільки буде два плюс два?", "file": "lab9/ghost.ipynb"},
                         {"n": 3, "text": "Яка столиця Франції та коли її заснували?", "file": None}],
           "sections": [], "media": []}
    spec, st = ai_steps.verify_spec(raw, TEXT)
    assert [r["path"] for r in spec["required"]] == ["lab9/a.ipynb"] and st["required_dropped"] == 1
    assert [f["kind"] for f in spec["forbidden"]] == ["model_weights"]
    assert [q["n"] for q in spec["questions"]] == [1, 2] and st["questions_dropped"] == 1
    assert spec["questions"][1]["text"].startswith("Скільки буде два плюс два і чому")  # перефразування «виправлено» за оригіналом
    assert spec["questions"][1]["file"] is None and st["files_nulled"] == 1
    assert spec["deadline"] == "2027-01-01"


def test_verify_spec_rejects_wrong_deadline():
    spec, _ = ai_steps.verify_spec({"required": [{"path": "lab9/a.ipynb", "type": "file", "quote": ""}], "deadline": "2027-02-02"}, TEXT)
    assert spec["deadline"] is None


def _st(n, words=40, code=0, block="### Питання\nПовна змістовна відповідь на це питання з конкретним прикладом і поясненням чому так."):
    return QuestionStatus(n, "f", f"питання {n}", True, "h", words, code, block)


class Fake:
    """Підставний клієнт: повертає заздалегідь задані відповіді."""
    def __init__(self, *texts):
        self.texts, self.stats, self.calls = list(texts), LLMStats(), 0

    def chat(self, system, user, max_tokens=0):
        self.calls += 1
        t = self.texts.pop(0)
        if isinstance(t, Exception):
            raise t
        return LLMResult(t, 10, 5)


def test_judge_rejects_verdict_without_verifiable_quote():
    s1, s2 = _st(1), _st(2)
    fake = Fake(json.dumps({"results": [
        {"n": 1, "verdict": "answered", "quote": "конкретним прикладом і поясненням", "reason": "ок"},
        {"n": 2, "verdict": "answered", "quote": "цієї цитати в тексті немає взагалі", "reason": "ок"}]}))
    ai_steps.judge_questions(fake, [s1, s2])
    assert s1.verdict == "answered" and s2.verdict == "needs_review"


def test_judge_quote_from_heading_is_not_accepted():
    s = _st(1, block="### Питання 1. Чому небо блакитне взагалі\nВідповідь про розсіювання світла в атмосфері, яка має достатню довжину для перевірки.")
    ai_steps.judge_questions(Fake(json.dumps({"results": [{"n": 1, "verdict": "answered", "quote": "Чому небо блакитне взагалі", "reason": ""}]})), [s])
    assert s.verdict == "needs_review"


def test_judge_skips_ai_for_empty_and_thin_answers():
    a, b = _st(1, words=3), QuestionStatus(2, "f", "q", found=False)
    fake = Fake()
    cnt = ai_steps.judge_questions(fake, [a, b])
    assert fake.calls == 0 and a.verdict == b.verdict == "missing" and cnt["skipped_thin"] == 1 and cnt["skipped_not_found"] == 1


def test_judge_survives_ai_outage_and_garbage():
    s1, s2 = _st(1), _st(2)
    ai_steps.judge_questions(Fake(LLMUnavailable("down")), [s1])
    assert s1.verdict == "needs_review"
    ai_steps.judge_questions(Fake("не json", "знову не json"), [s2])
    assert s2.verdict == "needs_review"


def test_json_repair_is_attempted_once():
    s = _st(1)
    good = json.dumps({"results": [{"n": 1, "verdict": "missing", "quote": "", "reason": "порожньо"}]})
    fake = Fake("Ось відповідь без json", good)
    ai_steps.judge_questions(fake, [s])
    assert fake.calls == 2 and s.verdict == "missing"


def test_unknown_verdict_becomes_needs_review():
    s = _st(1)
    ai_steps.judge_questions(Fake(json.dumps({"results": [{"n": 1, "verdict": "perfect", "quote": "", "reason": ""}]})), [s])
    assert s.verdict == "needs_review"


def test_section_titles_with_punctuation_in_assignment_are_kept():
    text = "Шаблон:\nКорпус і чому потрібен RAG: ...\nВисновок: ...\n/lab9/a.ipynb\n"
    raw = {"required": [{"path": "lab9/a.ipynb", "type": "file", "quote": ""}],
           "sections": [{"title": "Корпус і чому потрібен RAG", "file": "lab9/a.ipynb"}, {"title": "Вигадана секція про щось", "file": None}]}
    spec, st = ai_steps.verify_spec(raw, text)
    assert [s["title"] for s in spec["sections"]] == ["Корпус і чому потрібен RAG"] and st["sections_dropped"] == 1


# ---------- шляхи: модель пише коротко, код приводить до повної форми ----------
MKR_TEXT = (__import__("pathlib").Path(__file__).resolve().parents[1] / "inputs" / "mkr1_assignment.txt").read_text(encoding="utf-8")


def test_canonical_paths_from_assignment_blocks():
    c = ai_steps.canonical_paths(MKR_TEXT)
    assert c == {"mkr1/report.md": "file", "mkr1/solution": "dir", "mkr1/demo.mp4": "file", "mkr1/evidence": "dir"}


def test_verify_spec_resolves_short_and_wildcard_paths():
    """Саме те, що трапилось у реальному прогоні: модель дала report.md / solution/ / evidence/* без першої папки."""
    raw = {"required": [{"path": "report.md", "type": "file", "quote": ""}, {"path": "solution/", "type": "dir", "quote": ""},
                        {"path": "/mkr1/demo.mp4", "type": "file", "quote": ""}, {"path": "mkr1/evidence/*", "type": "dir", "quote": ""}],
           "media": [{"path": "demo.mp4", "min_s": 60, "max_s": 90, "quote": "Відео 60–90 секунд"}],
           "questions": [{"n": 1, "text": __import__("json").loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "specs" / "mkr1.json")
                                                             .read_text(encoding="utf-8"))["questions"][0]["text"], "file": "report.md"}]}
    spec, st = ai_steps.verify_spec(raw, MKR_TEXT)
    assert [(r["path"], r["type"]) for r in spec["required"]] == [
        ("mkr1/report.md", "file"), ("mkr1/solution", "dir"), ("mkr1/demo.mp4", "file"), ("mkr1/evidence", "dir")]
    assert st["required_dropped"] == 0 and st["paths_normalized"] == 4
    assert spec["media"][0]["path"] == "mkr1/demo.mp4"
    assert spec["questions"][0]["file"] == "mkr1/report.md" and st["files_nulled"] == 0


def test_verify_spec_dedupes_paths_after_normalisation():
    raw = {"required": [{"path": "mkr1/solution/*", "type": "dir"}, {"path": "solution/", "type": "dir"}, {"path": "mkr1/report.md", "type": "file"}]}
    spec, st = ai_steps.verify_spec(raw, MKR_TEXT)
    assert [r["path"] for r in spec["required"]] == ["mkr1/solution", "mkr1/report.md"] and st["required_duplicates"] == 1


def test_resolve_path_ambiguous_or_unknown_is_none():
    known = {"a/x.py": "file", "b/x.py": "file"}
    assert ai_steps.resolve_path("x.py", known) is None      # неоднозначно
    assert ai_steps.resolve_path("zzz.py", known) is None    # немає
    assert ai_steps.resolve_path("a/x.py", known) == "a/x.py"


def test_compare_specs_ignores_path_form():
    from labcheck import bench
    from labcheck.spec import load_spec
    gold = load_spec(bench.SOLUTION / "specs" / "mkr1.json")
    got = json.loads(json.dumps(gold))
    for r in got["required"]:
        r["path"] = "/" + r["path"] + ("/*" if r["type"] == "dir" else "")
    for q in got["questions"]:
        q["file"] = "/" + q["file"]
    m = bench.compare_specs(got, gold)
    assert m["required_precision"] == m["required_recall"] == m["question_file_ok"] == 1.0


# ---------- терплячі повтори ----------
def test_default_retries_survive_a_long_503_streak():
    s = Server([(503, {})] * 6 + [OK])
    try:
        assert client(s.url).chat("s", "u").text == '{"a": 1}' and s.calls == 7
    finally:
        s.close()


def test_llm_retries_from_env_and_bad_value_falls_back():
    base = {"LLM_API_KEY": "k", "LLM_MODEL": "m"}
    assert LLMClient.from_env({**base, "LLM_RETRIES": "9"}).retries == 9
    assert LLMClient.from_env({**base, "LLM_RETRIES": "abc"}).retries == 6
    assert LLMClient.from_env({**base, "LLM_RETRIES": "999"}).retries == 12
    assert LLMClient.from_env(base).retries == 6


def test_backoff_is_capped_and_grows():
    from labcheck.llm import _backoff
    assert 2 <= _backoff(0) <= 3.5 and 45 <= _backoff(9) <= 46.5


# ---------- повтор розбору після збою сервісу ----------
def test_spec_bench_retries_unavailable_extraction_after_pause():
    from labcheck import bench
    from labcheck.spec import load_spec
    gold = load_spec(bench.SOLUTION / "specs" / "mkr1.json")
    raw = json.dumps({"title": gold["title"], "deadline": gold["deadline"], "required": gold["required"], "forbidden": gold["forbidden"],
                      "media": gold["media"], "questions": gold["questions"], "sections": gold["sections"]}, ensure_ascii=False)
    fake = Fake(LLMUnavailable("HTTP 503"), raw)
    pauses = []
    out = bench.run_ai_spec_bench(fake, names=("mkr1",), passes=3, pause_s=7, sleep=pauses.append)
    assert out["mkr1"]["ok"] and out["mkr1"]["attempts"] == 2 and pauses == [7]
    assert out["mkr1"]["required_recall"] == 1.0


def test_spec_bench_does_not_retry_auth_or_format_errors():
    from labcheck import bench
    pauses = []
    out = bench.run_ai_spec_bench(Fake(LLMAuthError("HTTP 401")), names=("mkr1",), passes=3, sleep=pauses.append)
    assert not out["mkr1"]["ok"] and pauses == []
    out = bench.run_ai_spec_bench(Fake("не json", "знову не json"), names=("mkr1",), passes=3, sleep=pauses.append)
    assert not out["mkr1"]["ok"] and pauses == []


def test_spec_bench_gives_up_after_all_passes():
    from labcheck import bench
    pauses = []
    out = bench.run_ai_spec_bench(Fake(*[LLMUnavailable("HTTP 503")] * 3), names=("mkr1",), passes=3, pause_s=1, sleep=pauses.append)
    assert not out["mkr1"]["ok"] and out["mkr1"]["attempts"] == 3 and len(pauses) == 2


# ---------- зіставлення цитат: толерантне до форми, суворе до змісту ----------
RAW = "Запускаю **`python demo_run.py`** (один рядок), а потім дивлюсь на підсумок: «ГОТОВО» чи «НЕ ГОТОВО» до здачі, і ще — що знайдено."


@pytest.mark.parametrize("quote", [
    "python demo_run.py (один рядок)",                  # дужки й markdown: у тексті після очищення були зайві пробіли
    "Запускаю python demo_run.py",                       # без зірочок і зворотних лапок
    "потім дивлюсь на підсумок «ГОТОВО» чи «НЕ ГОТОВО»",  # інша пунктуація
    "ПІДСУМОК готово чи не готово до здачі",             # регістр
    "Запускаю python ... чи НЕ ГОТОВО",                  # фрагменти з пропуском, у порядку
    "Запускаю python… що знайдено",                      # «…» як символ
])
def test_quote_in_accepts_formatting_differences(quote):
    assert ai_steps.quote_in(quote, RAW)


@pytest.mark.parametrize("quote", [
    "Запускаю quantum_runner (один рядок)",              # вигадане слово
    "один рядок python demo_run.py",                      # слова в іншому порядку
    "Запускаю demo_run.py (один рядок)",                  # пропущено слово python
    "чи НЕ ГОТОВО ... Запускаю python",                   # фрагменти не в порядку
    "готово",                                              # закоротка: нічого не доводить
    "",
])
def test_quote_in_rejects_invented_or_reordered(quote):
    assert not ai_steps.quote_in(quote, RAW)


def test_rejected_quote_is_kept_for_diagnosis_and_shown_in_warning():
    from labcheck.report import ai_warnings
    s = _st(1)
    ai_steps.apply_verdicts([s], [{"n": 1, "verdict": "answered", "quote": "цієї фрази в тексті немає зовсім", "reason": "r"}])
    assert s.verdict == "needs_review" and s.rejected_quote.startswith("цієї фрази")

    class Rep:  # мінімальна заглушка CheckReport для ai_warnings
        class res:  # noqa: N801
            questions = [s]
    w = ai_warnings(Rep)
    assert "цієї фрази в тексті немає" in w[0][2]


def test_judge_bench_reports_request_latency_separately_from_wall_time(tmp_path):
    """Паузи між повторами не повинні потрапляти в «час AI» — він береться з часу самих запитів."""
    from labcheck import bench

    class Slow(Fake):
        def chat(self, system, user, max_tokens=0):
            r = super().chat(system, user, max_tokens)
            self.stats.latency_s += 2.0   # «час запиту» 2 с на пакет
            return r

    items = bench.build_judge_items(json.loads((bench.SOLUTION / "bench_data" / "lab6_answers.json").read_text(encoding="utf-8"))["items"])
    batches = -(-len(items) // 5)
    fake = Slow(*[json.dumps({"results": [{"n": k, "verdict": "missing", "quote": "", "reason": "x"} for k in range(1, 51)]})] * batches)
    out = bench.run_ai_judge_bench(fake)
    assert out["request_latency_s"] == pytest.approx(2.0 * batches) and out["latency_s"] < 5


def test_compare_specs_lists_missing_sections():
    from labcheck import bench
    from labcheck.spec import load_spec
    gold = load_spec(bench.SOLUTION / "specs" / "mkr1.json")
    got = json.loads(json.dumps(gold))
    dropped = got["sections"].pop()
    m = bench.compare_specs(got, gold)
    assert m["sections_missing"] == [dropped["title"]] and m["sections_recall"] < 1

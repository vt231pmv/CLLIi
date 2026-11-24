"""Мінімальний клієнт до будь-якого OpenAI-сумісного /chat/completions (Gemini, OpenRouter, Groq, Ollama, LM Studio...).

Без залежностей (лише stdlib). Є: терплячі повтори на 429/5xx (безкоштовні моделі часто «перевантажені» хвилинами),
дисковий кеш, облік токенів і вартості.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class LLMUnavailable(Exception):
    """AI недоступний (мережа, ліміти, ключ). Програма має продовжити без AI-кроків."""


class LLMAuthError(LLMUnavailable):
    pass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    failures: int = 0
    log: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("calls", "cache_hits", "prompt_tokens", "completion_tokens", "failures")}
        d["cost_usd"] = round(self.cost_usd, 6)
        d["latency_s"] = round(self.latency_s, 2)
        return d


def load_dotenv(path: Path | None = None, env: dict | None = None) -> None:
    """Дуже проста підтримка .env (KEY=VALUE), щоб не вимагати python-dotenv. Існуючі змінні не перезаписує."""
    env = os.environ if env is None else env
    for p in ([path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]):
        if p and p.is_file():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 90, retries: int = 6,
                 cache_dir: Path | str | None = None, price_in: float = 0.0, price_out: float = 0.0,
                 json_mode: bool = False, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.model = api_key, model
        self.timeout, self.retries = timeout, retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.price_in, self.price_out = price_in, price_out
        self.json_mode = json_mode
        self._sleep = sleep
        self.stats = LLMStats()

    @classmethod
    def from_env(cls, env: dict | None = None, cache_dir: Path | str | None = None) -> "LLMClient | None":
        env = os.environ if env is None else env
        key, model = env.get("LLM_API_KEY", ""), env.get("LLM_MODEL", "")
        if not key or not model:
            return None
        try:
            retries = max(0, min(int(env.get("LLM_RETRIES", "") or 6), 12))
        except ValueError:
            retries = 6
        return cls(env.get("LLM_BASE_URL", DEFAULT_BASE_URL), key, model, cache_dir=cache_dir, retries=retries,
                   price_in=float(env.get("LLM_PRICE_IN_PER_M", 0) or 0), price_out=float(env.get("LLM_PRICE_OUT_PER_M", 0) or 0),
                   json_mode=env.get("LLM_JSON_MODE", "") in ("1", "true", "yes"))

    # ----- кеш -----
    def _key(self, system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([self.model, system, user, max_tokens], ensure_ascii=False).encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMResult | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return LLMResult(d["text"], d.get("prompt_tokens", 0), d.get("completion_tokens", 0), True, d.get("latency_s", 0.0))
        except (OSError, ValueError, KeyError):
            return None  # відсутній або пошкоджений кеш = просто йдемо в мережу

    def _cache_put(self, key: str, r: LLMResult) -> None:
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(
                {"text": r.text, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                 "latency_s": r.latency_s, "model": self.model}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # ----- запит -----
    def chat(self, system: str, user: str, max_tokens: int = 4096) -> LLMResult:
        key = self._key(system, user, max_tokens)
        hit = self._cache_get(key)
        if hit is None:
            hit = self._request(system, user, max_tokens)
            self._cache_put(key, hit)
            self.stats.calls += 1
        else:
            self.stats.cache_hits += 1
        self.stats.prompt_tokens += hit.prompt_tokens
        self.stats.completion_tokens += hit.completion_tokens
        self.stats.cost_usd += (hit.prompt_tokens * self.price_in + hit.completion_tokens * self.price_out) / 1e6
        self.stats.latency_s += hit.latency_s
        self.stats.log.append({"prompt_tokens": hit.prompt_tokens, "completion_tokens": hit.completion_tokens,
                               "cached": hit.cached, "latency_s": round(hit.latency_s, 2)})
        return hit

    def _request(self, system: str, user: str, max_tokens: int) -> LLMResult:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode("utf-8")
        last = ""
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "User-Agent": "labcheck/1.0"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                dt = time.perf_counter() - t0
                text = payload["choices"][0]["message"]["content"] or ""
                u = payload.get("usage") or {}
                return LLMResult(text, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), False, dt)
            except urllib.error.HTTPError as e:
                code = e.code
                if code in (401, 403):
                    self.stats.failures += 1
                    raise LLMAuthError(f"HTTP {code}: ключ не прийнято або немає доступу до моделі «{self.model}»") from None
                if code == 404:
                    self.stats.failures += 1
                    raise LLMUnavailable(f"HTTP 404: перевірте LLM_BASE_URL і назву моделі «{self.model}» (команда doctor покаже доступні)") from None
                if code in (408, 409, 425, 429, 500, 502, 503, 504):
                    last = f"HTTP {code}"
                    wait = _retry_after(e) or _backoff(attempt)
                else:
                    self.stats.failures += 1
                    raise LLMUnavailable(f"HTTP {code}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = type(e).__name__
                wait = _backoff(attempt)
            except (KeyError, IndexError, ValueError):
                last = "неочікувана відповідь сервера"
                wait = 1
            if attempt < self.retries:
                self._sleep(wait)
        self.stats.failures += 1
        raise LLMUnavailable(f"сервіс недоступний після {self.retries + 1} спроб ({last})")

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "labcheck/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                d = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMAuthError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LLMUnavailable(type(e).__name__) from None
        return sorted(str(m.get("id", "")).replace("models/", "") for m in d.get("data", []))


def _backoff(attempt: int) -> float:
    """2, 4, 8, 16, 32, 45, 45... секунд (+ невеликий розкид, щоб кілька запусків не бились в сервер синхронно)."""
    return min(2 ** (attempt + 1), 45) + random.uniform(0, 1.5)


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    v = e.headers.get("Retry-After") if e.headers else None
    try:
        return min(float(v), 60.0) if v else None
    except ValueError:
        return None


def parse_json_loose(text: str):
    """Дістає JSON із відповіді моделі: прибирає ```-огорожі й текст навколо. Кидає ValueError, якщо не вийшло."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except ValueError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        return json.loads(t[a:b + 1])
    raise ValueError("у відповіді немає JSON")

# src/llm_client.py

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
from google.auth.transport.requests import Request
from google.oauth2 import id_token


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str
    generate_path: str = "/api/generate"
    model: str = "llama3"
    timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "1800"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "8"))

    token_refresh_skew_s: int = int(os.getenv("LLM_TOKEN_REFRESH_SKEW_S", "120"))
    allow_manual_token: bool = os.getenv("LLM_ALLOW_MANUAL_TOKEN", "1") == "1"

    # Generation controls
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "220"))
    # Comma-separated stop sequences (kept simple)
    stop_csv: str = os.getenv("LLM_STOP", "```,")


class _IdTokenProvider:
    def __init__(self, audience: str, refresh_skew_s: int, allow_manual_token: bool):
        self._audience = audience.rstrip("/")
        self._refresh_skew_s = max(0, int(refresh_skew_s))
        self._allow_manual = bool(allow_manual_token)

        self._req = Request()
        self._cached_token: Optional[str] = None
        self._cached_exp: int = 0

    @staticmethod
    def _jwt_exp_unverified(token: str) -> int:
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return int(payload.get("exp", 0))
        except Exception:
            return 0

    def _manual_token(self) -> Optional[str]:
        manual = os.getenv("MANUAL_ID_TOKEN")
        if not manual:
            return None
        manual = manual.strip()
        return manual if manual else None

    def _mint_token(self) -> str:
        return id_token.fetch_id_token(self._req, self._audience)

    def get(self) -> str:
        now = int(time.time())

        if self._cached_token and self._cached_exp:
            if (now + self._refresh_skew_s) < self._cached_exp:
                return self._cached_token

        manual = self._manual_token()
        if manual is not None:
            if not self._allow_manual:
                raise RuntimeError(
                    "MANUAL_ID_TOKEN is set but manual tokens are disabled. "
                    "Unset MANUAL_ID_TOKEN or set LLM_ALLOW_MANUAL_TOKEN=1."
                )

            exp = self._jwt_exp_unverified(manual)
            if exp and (now + self._refresh_skew_s) >= exp:
                raise RuntimeError(
                    "MANUAL_ID_TOKEN has expired (or is too close to expiry). "
                    "For long runs, do NOT use a static token. "
                    "Either run on Cloud Run (preferred) or re-mint a fresh token locally."
                )

            self._cached_token = manual
            self._cached_exp = exp or 0
            return manual

        try:
            tok = self._mint_token()
            exp = self._jwt_exp_unverified(tok)
            self._cached_token = tok
            self._cached_exp = exp or (now + 3000)
            return tok
        except Exception as e:
            raise RuntimeError(
                "Could not mint an OIDC ID token.\n\n"
                "Local option:\n"
                "  export MANUAL_ID_TOKEN=$(gcloud auth print-identity-token "
                f'--audiences="{self._audience}")\n\n'
                "GCP option:\n"
                "  Run on Cloud Run/Compute with a service account that has roles/run.invoker\n"
                "  on the target service, and let ADC mint tokens automatically.\n"
            ) from e

    def force_refresh(self) -> str:
        self._cached_token = None
        self._cached_exp = 0
        return self.get()


class LLMClient:
    LABELS = {"YES", "LEAN_YES", "UNSURE", "LEAN_NO", "NO"}

    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._audience = cfg.base_url.rstrip("/")
        self._token_provider = _IdTokenProvider(
            audience=self._audience,
            refresh_skew_s=cfg.token_refresh_skew_s,
            allow_manual_token=cfg.allow_manual_token,
        )

    @staticmethod
    def _strip_known_prefixes(s: str) -> str:
        s = (s or "").strip()
        for prefix in (
            "Here is the JSON output:",
            "Here is the JSON response:",
            "Here is the output:",
            "Here is the output in JSON format:",
            "JSON:",
        ):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        return s

    @staticmethod
    def _extract_braced_block(s: str) -> Optional[str]:
        if not s:
            return None

        # Prefer fenced JSON block if present
        if "```" in s:
            parts = s.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    return p

        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return s[start : end + 1]

    @classmethod
    def _sanitize_json(cls, js: str) -> str:
        """
        Fix common LLM JSON mistakes:
        - Unquoted label tokens: "label": LEAN_NO  -> "label": "LEAN_NO"
        - Unquoted YES/NO variants
        """
        if not js:
            return js

        # Quote label values if they look like bare tokens
        def repl_label(m: re.Match) -> str:
            token = m.group(1)
            if token in cls.LABELS:
                return f'"label":"{token}"'
            return m.group(0)

        js = re.sub(r'"label"\s*:\s*([A-Z_]+)', repl_label, js)

        # Sometimes confidence gets quoted; that's valid JSON but you want float later
        return js

    @classmethod
    def _try_parse_json(cls, raw_text: str) -> Optional[Dict[str, Any]]:
        s = cls._strip_known_prefixes(raw_text)
        block = cls._extract_braced_block(s)
        if not block:
            return None
        block = cls._sanitize_json(block)
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    @classmethod
    def _salvage_fields(cls, raw_text: str) -> Dict[str, Any]:
        """
        As a fallback when strict JSON parsing fails, try to extract:
        label, confidence, rationale
        """
        s = raw_text or ""
        # label
        m_label = re.search(r'"label"\s*:\s*"?([A-Z_]+)"?', s)
        label = m_label.group(1) if m_label and m_label.group(1) in cls.LABELS else None

        # confidence
        m_conf = re.search(r'"confidence"\s*:\s*"?([0-9]*\.?[0-9]+)"?', s)
        conf = None
        if m_conf:
            try:
                conf = float(m_conf.group(1))
            except Exception:
                conf = None

        # rationale (capture JSON string value if present)
        m_rat = re.search(r'"rationale"\s*:\s*"(.*?)"\s*(?:,|\n|\})', s, flags=re.DOTALL)
        rationale = None
        if m_rat:
            rationale = m_rat.group(1).replace('\\"', '"').strip()

        return {
            "label": label,
            "confidence": conf,
            "rationale": rationale,
            "salvaged": True,
        }

    def _stop_list(self) -> list[str]:
        # User-configured stops + a couple of safe defaults to prevent post-JSON chatter
        raw = self.cfg.stop_csv or ""
        stops = [x.strip() for x in raw.split(",") if x.strip()]
        # Add common offenders (don’t duplicate)
        for s in ["\nRationale:", "\n\nRationale:", "```"]:
            if s not in stops:
                stops.append(s)
        return stops

    async def infer_json(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_path

        payload = {
            "model": self.cfg.model,
            "prompt": prompt_text,
            "stream": False,
            # If supported by your Ollama build, this strongly nudges “JSON only”.
            "format": "json",
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
                "stop": self._stop_list(),
            },
        }

        last_err: Optional[Exception] = None
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
        retryable = {429, 500, 502, 503, 504}

        token = self._token_provider.get()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    raw_body = await resp.text()

                    if resp.status in (401, 403):
                        last_err = RuntimeError(f"Ollama HTTP {resp.status}: {raw_body[:800]}")
                        token = self._token_provider.force_refresh()
                        headers["Authorization"] = f"Bearer {token}"
                        await asyncio.sleep(min(2 ** (attempt - 1), 10))
                        continue

                    if resp.status in retryable:
                        ra = resp.headers.get("Retry-After")
                        delay: Optional[float] = None
                        if ra:
                            try:
                                delay = float(ra)
                            except ValueError:
                                delay = None
                        if delay is None:
                            base = min(2 ** (attempt - 1), 60)
                            jitter = random.uniform(0, base * 0.3)
                            delay = base + jitter
                        if resp.status == 503:
                            delay = max(delay, 20.0)

                        last_err = RuntimeError(f"Ollama HTTP {resp.status}: {raw_body[:800]}")
                        print(f"[llm] retry attempt={attempt}/{self.cfg.max_retries} status={resp.status} delay={delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue

                    if resp.status >= 400:
                        raise RuntimeError(f"Ollama HTTP {resp.status}: {raw_body[:800]}")

                    # Ollama returns JSON with a "response" field (string)
                    try:
                        data = json.loads(raw_body) if raw_body else {}
                    except json.JSONDecodeError:
                        data = {}

                    if isinstance(data, dict):
                        raw_text = (data.get("response") or "").strip() or raw_body.strip()
                        meta = {
                            "done": data.get("done"),
                            "done_reason": data.get("done_reason"),
                            "eval_count": data.get("eval_count"),
                            "prompt_eval_count": data.get("prompt_eval_count"),
                            "total_duration": data.get("total_duration"),
                        }
                    else:
                        raw_text = raw_body.strip()
                        meta = {}

                    parsed = self._try_parse_json(raw_text)
                    if parsed is not None:
                        return parsed

                    salvaged = self._salvage_fields(raw_text)
                    return {
                        "parse_error": True,
                        "raw_response": raw_text,
                        "ollama_meta": meta,
                        **salvaged,
                    }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                base = min(2 ** (attempt - 1), 60)
                jitter = random.uniform(0, base * 0.3)
                delay = base + jitter
                print(f"[llm] retry attempt={attempt}/{self.cfg.max_retries} error={type(e).__name__} delay={delay:.1f}s")
                await asyncio.sleep(delay)

            except Exception as e:
                last_err = e
                base = min(2 ** (attempt - 1), 60)
                jitter = random.uniform(0, base * 0.3)
                delay = base + jitter
                print(f"[llm] retry attempt={attempt}/{self.cfg.max_retries} error={type(e).__name__} delay={delay:.1f}s")
                await asyncio.sleep(delay)

        raise RuntimeError(f"LLM request failed after retries: {last_err}")
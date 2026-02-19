from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp
from google.auth.transport.requests import Request
from google.oauth2 import id_token


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str

    #keep this for backwards compatibility, if you wanna switch to generate instead of chat endpoint
    generate_path: str = "/api/chat"

    model: str = "llama3:8b"
    timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "1800"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "8"))

    token_refresh_skew_s: int = int(os.getenv("LLM_TOKEN_REFRESH_SKEW_S", "120"))
    allow_manual_token: bool = os.getenv("LLM_ALLOW_MANUAL_TOKEN", "1") == "1"

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1")) #zero is total determinism
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9")) #smallest set of tokens that fit probability p, which filters out low prob tokens
    num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "220")) #token output restriction
    stop_csv: str = os.getenv("LLM_STOP", "```,")

    #set seed for reproducibility if you want
    seed: Optional[int] = int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else None


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
    EVIDENCE_GRADES = {"A", "B", "C", "D", "F"}

    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._audience = cfg.base_url.rstrip("/")
        self._token_provider = _IdTokenProvider(
            audience=self._audience,
            refresh_skew_s=cfg.token_refresh_skew_s,
            allow_manual_token=cfg.allow_manual_token,
        )

    # ----------------------------
    # Structured output schema
    # ----------------------------
    @classmethod
    def _output_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": sorted(list(cls.LABELS))},
                "evidence_grade": {"type": "string", "enum": sorted(list(cls.EVIDENCE_GRADES))},
                "rationale": {"type": "string"},
                "model_version": {"type": "string"},
                "prompt_version": {"type": "string"},
            },
            "required": ["label", "evidence_grade", "rationale", "model_version", "prompt_version"],
        }

    # ----------------------------
    # Parsing helpers
    # ----------------------------
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
                s = s[len(prefix) :].strip()
        return s

    @staticmethod
    def _extract_braced_block(s: str) -> Optional[str]:
        if not s:
            return None

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
        #hopefully this helps fix output not respecting json format
        if not js:
            return js

        def repl_label(m: re.Match) -> str:
            token = m.group(1)
            if token in cls.LABELS:
                return f'"label":"{token}"'
            return m.group(0)

        def repl_grade(m: re.Match) -> str:
            token = m.group(1)
            if token in cls.EVIDENCE_GRADES:
                return f'"evidence_grade":"{token}"'
            return m.group(0)

        js = re.sub(r'"label"\s*:\s*([A-Z_]+)', repl_label, js)
        js = re.sub(r'"evidence_grade"\s*:\s*([A-Z])', repl_grade, js)
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
        s = raw_text or ""

        m_label = re.search(r'"label"\s*:\s*"?([A-Z_]+)"?', s)
        label = m_label.group(1) if m_label and m_label.group(1) in cls.LABELS else None

        m_grade = re.search(r'"evidence_grade"\s*:\s*"?([A-Z])"?', s)
        evidence_grade = (
            m_grade.group(1) if m_grade and m_grade.group(1) in cls.EVIDENCE_GRADES else None
        )

        m_rat = re.search(r'"rationale"\s*:\s*"(.*?)"\s*(?:,|\n|\})', s, flags=re.DOTALL)
        rationale = None
        if m_rat:
            rationale = m_rat.group(1).replace('\\"', '"').strip()

        return {"label": label, "evidence_grade": evidence_grade, "rationale": rationale, "salvaged": True}

    def _stop_list(self) -> list[str]:
        raw = self.cfg.stop_csv or ""
        stops = [x.strip() for x in raw.split(",") if x.strip()]
        for s in ["\nRationale:", "\n\nRationale:", "```"]:
            if s not in stops:
                stops.append(s)
        return stops

    
    def _endpoint_mode(self) -> str:
        path = (self.cfg.generate_path or "").strip()
        if path.endswith("/api/chat"):
            return "chat"
        if path.endswith("/api/generate"):
            return "generate"
        return "chat"

    def _build_request(self, prompt_text: str) -> Tuple[str, Dict[str, Any], str]:
        url = self.cfg.base_url.rstrip("/") + (self.cfg.generate_path or "/api/chat")
        mode = self._endpoint_mode()

        options: Dict[str, Any] = {
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "num_predict": int(self.cfg.num_predict),
            "stop": self._stop_list(),
        }
        if self.cfg.seed is not None:
            options["seed"] = int(self.cfg.seed)

        if mode == "chat":
            payload = {
                "model": self.cfg.model,
                "stream": False,
                "format": self._output_schema(),
                "messages": [{"role": "user", "content": prompt_text}],
                "options": options,
            }
            return url, payload, mode

        payload = {
            "model": self.cfg.model,
            "prompt": prompt_text,
            "stream": False,
            "format": self._output_schema(),
            "options": options,
        }
        return url, payload, mode

    def _extract_text_and_meta(self, raw_body: str, mode: str) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        raw_text = raw_body.strip()

        try:
            data = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            data = {}

        if isinstance(data, dict):
            meta = {
                "done": data.get("done"),
                "done_reason": data.get("done_reason"),
                "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "total_duration": data.get("total_duration"),
            }

            if mode == "chat":
                msg = data.get("message") or {}
                if isinstance(msg, dict):
                    raw_text = (msg.get("content") or "").strip() or raw_text
            else:
                raw_text = (data.get("response") or "").strip() or raw_text

        return raw_text, meta

    async def infer_json(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict[str, Any]:
        url, payload, mode = self._build_request(prompt_text)

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
                        print(
                            f"[llm] retry attempt={attempt}/{self.cfg.max_retries} "
                            f"status={resp.status} delay={delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                    if resp.status >= 400:
                        raise RuntimeError(f"Ollama HTTP {resp.status}: {raw_body[:800]}")

                    raw_text, meta = self._extract_text_and_meta(raw_body, mode)

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
                print(
                    f"[llm] retry attempt={attempt}/{self.cfg.max_retries} "
                    f"error={type(e).__name__} delay={delay:.1f}s"
                )
                await asyncio.sleep(delay)

            except Exception as e:
                last_err = e
                base = min(2 ** (attempt - 1), 60)
                jitter = random.uniform(0, base * 0.3)
                delay = base + jitter
                print(
                    f"[llm] retry attempt={attempt}/{self.cfg.max_retries} "
                    f"error={type(e).__name__} delay={delay:.1f}s"
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"LLM request failed after retries: {last_err}")
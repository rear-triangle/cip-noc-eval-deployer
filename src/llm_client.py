# src/llm_client.py

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request
from google.oauth2 import id_token


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str                 # e.g. https://service.run.app
    generate_path: str = "/api/generate"
    model: str = "llama3"
    timeout_s: int = 300
    max_retries: int = 4

    # "oidc" (Cloud Run private) | "none" (public / no auth)
    auth_mode: str = "oidc"

    # Optional local-dev helper:
    # If set, we mint an audience-bound ID token by impersonating this SA
    impersonate_service_account: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._audience = cfg.base_url.rstrip("/")

    def _get_oidc_token(self) -> str:
        """
        Token selection order:

        0) If auth_mode == "none": no token
        1) If MANUAL_ID_TOKEN is provided (local smoke tests): use it
        2) If impersonate_service_account is set: mint via IAMCredentials (no gcloud required)
        3) Otherwise: try metadata-server minting (works on Cloud Run/GCE)
        """
        if self.cfg.auth_mode == "none":
            return ""

        # 1) Local override (what you already proved works)
        manual = os.getenv("MANUAL_ID_TOKEN")
        if manual:
            return manual.strip()

        req = Request()

        # 2) Impersonation path (no need for gcloud in container)
        if self.cfg.impersonate_service_account:
            source_creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            id_creds = impersonated_credentials.IDTokenCredentials(
                source_credentials=source_creds,
                target_principal=self.cfg.impersonate_service_account,
                target_audience=self._audience,
                include_email=True,
            )
            id_creds.refresh(req)

            if not getattr(id_creds, "token", None):
                raise RuntimeError("Impersonated ID token credential returned no token.")
            return id_creds.token

        # 3) In-GCP environments (Cloud Run/Compute metadata server)
        try:
            return id_token.fetch_id_token(req, self._audience)
        except Exception as e:
            raise RuntimeError(
                "Could not mint an OIDC ID token.\n\n"
                "Local options:\n"
                "  A) Export MANUAL_ID_TOKEN on your host and pass it into docker:\n"
                f'     MANUAL_ID_TOKEN=$(gcloud auth print-identity-token --audiences="{self._audience}" '
                "--impersonate-service-account=YOUR_SA@PROJECT.iam.gserviceaccount.com)\n"
                "  B) Set llm.impersonate_service_account in config (requires Token Creator)\n"
                '  C) Make the Cloud Run service public and set llm.auth_mode: "none"\n\n'
                "GCP option:\n"
                "  - Run this on Cloud Run where metadata server is available.\n"
            ) from e

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        s = text.strip()

        # Remove common prefixes
        for prefix in (
            "Here is the JSON output:",
            "Here is the JSON response:",
            "JSON:",
        ):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()

        # Handle fenced code blocks
        if "```" in s:
            parts = s.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    s = p
                    break

        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    async def infer_json(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_path

        headers = {"Content-Type": "application/json"}
        if self.cfg.auth_mode != "none":
            token = self._get_oidc_token()
            headers["Authorization"] = f"Bearer {token}"

        payload = {
            "model": self.cfg.model,
            "prompt": prompt_text,
            "stream": False,
        }

        last_err: Optional[Exception] = None
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise RuntimeError(f"Ollama HTTP {resp.status}: {json.dumps(data)[:800]}")

                    raw_text = (data.get("response") or "").strip()

                    parsed = self._extract_json_obj(raw_text)
                    if parsed is not None:
                        return parsed

                    return {"parse_error": True, "raw_response": raw_text, "ollama": data}

            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** (attempt - 1), 10))

        raise RuntimeError(f"LLM request failed after retries: {last_err}")
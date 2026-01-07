# src/llm_client.py

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

import google.auth
from google.auth.transport.requests import Request, AuthorizedSession
from google.oauth2 import id_token
from google.auth import impersonated_credentials


@dataclass(frozen=True)
class LLMClientConfig:
    # Cloud Run service base URL (NO PATH), e.g. https://ollama-gcs-llama3-gpu-xxxx.us-east4.run.app
    base_url: str

    # Ollama API endpoint path
    generate_path: str = "/api/generate"

    # Ollama model name
    model: str = "llama3"

    timeout_s: int = 300
    max_retries: int = 4

    # For local dev: service account email to impersonate and mint an ID token
    # Example: "228291553312-compute@developer.gserviceaccount.com"
    impersonate_service_account: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._audience = cfg.base_url.rstrip("/")

    def _get_oidc_token(self) -> str:
        """
        Mint an ID token suitable for invoking a Cloud Run service.

        1) On Cloud Run: uses metadata server via id_token.fetch_id_token(...)
        2) Locally (ADC): impersonates a service account and calls IAMCredentials generateIdToken
        """
        req = Request()

        # 1) Cloud Run / GCE metadata server path
        try:
            return id_token.fetch_id_token(req, self._audience)
        except Exception:
            pass

        # 2) Local dev path
        sa = self.cfg.impersonate_service_account
        if not sa:
            raise RuntimeError(
                "Could not mint ID token from default credentials. "
                "For local runs, set llm.impersonate_service_account in configs/dev.yaml "
                "and grant your user 'Service Account Token Creator' on that SA."
            )

        # Source creds: your local ADC (user creds) mounted into Docker
        source_creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Impersonate SA to get an *access token* that can call IAMCredentials APIs
        imp_creds = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=sa,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            lifetime=3600,
        )

        authed = AuthorizedSession(imp_creds)

        # Call IAMCredentials generateIdToken
        url = f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateIdToken"
        body = {"audience": self._audience, "includeEmail": True}

        resp = authed.post(url, json=body, timeout=30)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"IAMCredentials generateIdToken failed: HTTP {resp.status_code}: {resp.text[:1000]}"
            )

        payload = resp.json()
        token = payload.get("token")
        if not token:
            raise RuntimeError(f"generateIdToken response missing 'token': {payload}")

        return token

    async def infer_json(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict[str, Any]:
        """
        Calls Ollama /api/generate (non-streaming).
        Expects Ollama JSON; then parses the model's `response` field as JSON (per your prompt contract).
        """
        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_path
        token = self._get_oidc_token()

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"model": self.cfg.model, "prompt": prompt_text, "stream": False}

        last_err: Optional[Exception] = None
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise RuntimeError(f"Ollama HTTP {resp.status}: {json.dumps(data)[:800]}")

                    raw_text = (data.get("response") or "").strip()

                    # Your prompt demands JSON-only
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        return {"parse_error": True, "raw_response": raw_text, "ollama": data}

            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** (attempt - 1), 10))

        raise RuntimeError(f"LLM request failed after retries: {last_err}")

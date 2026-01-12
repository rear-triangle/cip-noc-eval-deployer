# src/llm_client.py

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
from google.auth.transport.requests import Request
from google.oauth2 import id_token


@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str                 # e.g. https://service.run.app
    generate_path: str = "/api/generate"
    model: str = "llama3"
    timeout_s: int = 300
    max_retries: int = 4

    # Auth mode:
    # - "oidc" (default): send Authorization: Bearer <ID_TOKEN>
    # - "none": do not send auth header (only works if your Cloud Run service is public)
    auth_mode: str = "oidc"

    # Optional local-dev helper:
    # If set, we can mint an ID token via gcloud impersonation:
    #   gcloud auth print-identity-token --impersonate-service-account=... --audiences=...
    impersonate_service_account: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMClientConfig):
        self.cfg = cfg
        self._audience = cfg.base_url.rstrip("/")

    def _get_oidc_token(self) -> Optional[str]:
        """
        Token order (most reliable first):

        1) MANUAL_ID_TOKEN env var (explicit override; matches your old working codebase)
        2) In GCP (Cloud Run/Compute): id_token.fetch_id_token() uses metadata server
        3) Local fallback via gcloud impersonation (if configured + gcloud exists)

        Note: User ADC credentials (application_default_credentials.json) typically cannot mint
        audience-bound ID tokens. That's why BigQuery works but fetch_id_token fails locally.
        """
        manual = os.getenv("MANUAL_ID_TOKEN")
        if manual:
            return manual.strip()

        req = Request()

        # Works in GCP environments (metadata server present)
        try:
            return id_token.fetch_id_token(req, self._audience)
        except Exception:
            pass

        # Local fallback via gcloud impersonation (only if configured and gcloud exists)
        if self.cfg.impersonate_service_account and shutil.which("gcloud"):
            cmd = [
                "gcloud",
                "auth",
                "print-identity-token",
                f"--impersonate-service-account={self.cfg.impersonate_service_account}",
                f"--audiences={self._audience}",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "Failed to mint ID token via gcloud.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{e.output}"
                ) from e

            if not out:
                raise RuntimeError("gcloud returned an empty identity token.")
            return out

        # If we got here, we don't have a way to mint a token in this environment
        raise RuntimeError(
            "Could not mint an OIDC ID token for Cloud Run.\n\n"
            "This commonly happens locally because ADC user credentials "
            "(application_default_credentials.json) can access BigQuery but cannot mint "
            "audience-bound ID tokens.\n\n"
            "Fix options:\n"
            "  A) (Recommended for local smoke tests) Export MANUAL_ID_TOKEN from your host:\n"
            "     MANUAL_ID_TOKEN=$(gcloud auth print-identity-token --audiences=\"{aud}\")\n"
            "     and pass -e MANUAL_ID_TOKEN into docker.\n\n"
            "  B) Set llm.impersonate_service_account in configs/dev.yaml and ensure gcloud exists in the container.\n"
            "  C) Make the LLM Cloud Run service public and set llm.auth_mode: \"none\".\n"
            .format(aud=self._audience)
        )

    async def infer_json(self, session: aiohttp.ClientSession, prompt_text: str) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + self.cfg.generate_path

        headers = {"Content-Type": "application/json"}
        if self.cfg.auth_mode.lower() != "none":
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

                    # Your prompt asks for JSON only, so parse it
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        return {"parse_error": True, "raw_response": raw_text, "ollama": data}

            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** (attempt - 1), 10))

        raise RuntimeError(f"LLM request failed after retries: {last_err}")
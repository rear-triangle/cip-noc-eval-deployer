from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@dataclass(frozen=True)
class Prompt:
    version: str
    template_text: str

def load_prompt(prompt_path: str, prompt_version: str) -> Prompt:
    txt = Path(prompt_path).read_text(encoding="utf-8")
    return Prompt(version=prompt_version, template_text=txt)

def render_prompt(prompt: Prompt, fields: Dict[str, Any]) -> str:
    # Super simple templating: {{key}} replacement.
    # If you prefer Jinja2 later, swap this out.
    rendered = prompt.template_text.replace("{{prompt_version}}", prompt.version)
    for k, v in fields.items():
        token = "{{" + k + "}}"
        rendered = rendered.replace(token, "" if v is None else str(v))
    return rendered

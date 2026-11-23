"""LLM clients: Ollama (default), any OpenAI-compatible API, and an offline mock.

Configuration comes from environment variables (or a ``.env`` file)::

    LLM_PROVIDER = ollama | openai | mock
    LLM_MODEL    = llama3.1:8b | qwen2.5:7b | gpt-4o-mini | ...
    LLM_API_KEY  = ...          (openai only)
    LLM_BASE_URL = http://localhost:11434  (ollama)  |  https://api.openai.com/v1  (openai)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol

try:  # optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

#: markers shared with :mod:`src.extract` so that the mock can read the prompt
DOC_START, DOC_END = "<<<DOCUMENT", "DOCUMENT>>>"
GUIDED_MARKER = "ALLOWED VOCABULARY"


class LLMClient(Protocol):
    """Minimal chat-completion interface."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the raw text answer of the model."""
        ...


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from config)
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class OllamaClient:
    """Local Ollama server (``ollama serve``), JSON mode, temperature 0."""

    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, system: str, user: str) -> str:
        data = _post_json(
            f"{self.base_url.rstrip('/')}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 8192},
            },
        )
        return data["message"]["content"]


@dataclass
class OpenAICompatibleClient:
    """Any server implementing ``POST /chat/completions`` (OpenAI, Mistral, vLLM, LM Studio...)."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, user: str) -> str:
        data = _post_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- mock

# (regex, field, ontology id, free-text name a model would plausibly use without guidance)
_LEXICON: list[tuple[str, str, str, str]] = [
    # resource types
    (r"\baffiche\b|\bposter\b", "resource_type", "Poster", "poster"),
    (r"fiche pratique|practical sheet|fiche m[ée]mo|check-list", "resource_type", "PracticalSheet", "fact sheet"),
    (r"\bformation\b|\bMOOC\b|\batelier\b|\bcourse\b|\btraining\b", "resource_type", "Training", "course"),
    (r"\bvid[ée]o\b", "resource_type", "Video", "video"),
    (r"web application|\boutil\b|\btool\b", "resource_type", "Tool", "software"),
    # audiences
    (r"infirmi[eè]r|\bIDE\b|\bnurses?\b", "target_audience", "nurse", "Nurses"),
    (r"aides?-soignant|care assistants?", "target_audience", "care_assistant", "Nursing assistants"),
    (r"m[ée]decins?|physicians?|\bGP\b", "target_audience", "physician", "Doctors"),
    (r"pharmaciens?|pharmacists?", "target_audience", "pharmacist", "Pharmacists"),
    (r"\bfamilles?\b|\bproches\b|families", "target_audience", "family", "Relatives"),
    (r"aux r[ée]sidents|invites residents|r[ée]sidents eux-m[êe]mes", "target_audience", "resident", "Nursing home residents"),
    # topics
    (r"infections? urinaire|urinary tract infection|bact[ée]riurie|bacteriuria|cystite", "topic", "uti", "UTI"),
    (r"pneumopathie|pneumonia|infections? respiratoire|bronchite", "topic", "respiratory_infection", "Pneumonia"),
    (r"antibior[ée]sistance|r[ée]sistance aux antibiotiques|antimicrobial resistance|multir[ée]sistante|resistant bacteria", "topic", "antibiotic_resistance", "AMR"),
    (r"hygi[eè]ne des mains|friction hydro-alcoolique|hand rub", "topic", "hand_hygiene", "Hand hygiene"),
    (r"dur[ée]es? de traitement|r[ée][ée]valuation|r[ée][ée]valuer|prescribing", "topic", "prescription", "Antibiotic prescribing"),
    (r"\bvaccin|\bvaccine|vaccination", "topic", "vaccination", "Vaccination"),
    # objectives
    (r"sensibiliser|raise awareness", "objective", "raise_awareness", "Awareness"),
    (r"repérer|improve diagnosis|am[ée]liorer le diagnostic", "objective", "improve_diagnosis", "Better diagnosis"),
    (r"antibiotiques inutiles|unnecessary antibiotics", "objective", "reduce_unnecessary_prescription", "Antibiotic sparing"),
    (r"qualit[ée] des prescriptions|appropriate prescribing", "objective", "improve_prescription_quality", "Prescribing quality"),
    (r"pr[ée]venir les infections|prevent infections", "objective", "prevent_infection", "Infection prevention"),
    (r"former les professionnels|train staff", "objective", "train_staff", "Staff education"),
    # formats
    (r"\bA3\b|imprimable|printed", "format", "print", "Printed poster"),
    (r"\bPDF\b", "format", "pdf", "PDF"),
    (r"en ligne de trois modules|online course|\bMOOC\b", "format", "elearning", "Online course"),
    (r"pr[ée]sentiel", "format", "face_to_face", "Workshop"),
    (r"vid[ée]o en ligne|vid[ée]o de", "format", "video_file", "Online video"),
    (r"web application", "format", "web_app", "Web app"),
    # usage conditions
    (r"libre acc[eè]s|\bfree\b|gratuitement", "usage_condition", "free_access", "Free"),
    (r"inscription obligatoire|registration", "usage_condition", "registration_required", "Registration needed"),
    (r"formateur", "usage_condition", "trainer_required", "Trainer-led"),
    (r"creative commons", "usage_condition", "open_licence", "CC licence"),
    (r"professionals only|r[ée]serv[ée] aux professionnels", "usage_condition", "professional_use_only", "Professionals only"),
]
_SINGLE = {"resource_type", "format"}


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_candidates = [i for i in (text.find(".", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right].strip()


@dataclass
class MockClient:
    """Deterministic, offline stand-in for an LLM (rule-based keyword matcher).

    It is **not** a language model: it exists so that the whole pipeline, the tests and
    the CI run without a GPU or network. It reads the document from the prompt; when the
    prompt contains the ontology vocabulary it answers with ontology identifiers, otherwise
    with the free-text names a model typically produces without guidance.
    """

    name: str = "mock:keyword-baseline"

    def complete(self, system: str, user: str) -> str:
        guided = GUIDED_MARKER in user or GUIDED_MARKER in system
        text = user.split(DOC_START, 1)[-1].split(DOC_END, 1)[0].strip()
        title = text.splitlines()[0].strip()
        out: dict[str, object] = {
            "resource_type": None, "title": {"value": title, "excerpt": title}, "publisher": None,
            "publication_year": None, "language": None,
            "target_audience": [], "objective": [], "topic": [], "format": None, "usage_condition": [],
        }
        if m := re.search(r"\b(19|20)\d{2}\b", text):
            out["publication_year"] = {"value": int(m.group()), "excerpt": _sentence_around(text, m.start(), m.end())}
        fr_hits = len(re.findall(r"\b(les|des|est|une|pour|aux|et)\b", text, re.I))
        en_hits = len(re.findall(r"\b(the|and|is|for|of|to)\b", text, re.I))
        out["language"] = {"value": "fr" if fr_hits > en_hits else "en", "excerpt": ""}
        best_single: dict[str, tuple[int, dict]] = {}
        for pattern, fieldname, onto_id, free in _LEXICON:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            item = {"value": onto_id if guided else free, "excerpt": _sentence_around(text, m.start(), m.end())}
            if fieldname in _SINGLE:
                if fieldname not in best_single or m.start() < best_single[fieldname][0]:
                    best_single[fieldname] = (m.start(), item)
            else:
                values = out[fieldname]
                assert isinstance(values, list)
                if item["value"] not in [v["value"] for v in values]:
                    values.append(item)
        for fieldname, (_, item) in best_single.items():
            out[fieldname] = item
        return json.dumps(out, ensure_ascii=False)


def get_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Instantiate the client configured by arguments or environment variables."""
    if load_dotenv is not None:
        load_dotenv()
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    model = model or os.getenv("LLM_MODEL") or None
    base_url = os.getenv("LLM_BASE_URL") or None
    if provider == "mock":
        return MockClient()
    if provider == "ollama":
        if base_url and "/v1" in base_url:  # an OpenAI URL left in .env
            base_url = None
        return OllamaClient(model=model or "llama3.1:8b", base_url=base_url or "http://localhost:11434")
    if provider in {"openai", "openai-compatible"}:
        key = os.getenv("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is required for LLM_PROVIDER=openai")
        return OpenAICompatibleClient(model=model or "gpt-4o-mini", api_key=key,
                                      base_url=base_url or "https://api.openai.com/v1")
    raise ValueError(f"unknown LLM provider: {provider!r}")

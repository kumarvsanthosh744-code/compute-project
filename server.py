
"""ScamShield AI backend. Install fastapi uvicorn google-genai pydantic python-multipart."""
from __future__ import annotations

import logging
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("scamshield")
ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TEXT_CHARS = 20_000
MAX_AUDIO_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/ogg", "application/ogg",
    "audio/webm", "video/webm", "application/octet-stream",
}


class SignalScores(BaseModel):
    authority_impersonation: int = Field(ge=0, le=100)
    urgency_isolation: int = Field(ge=0, le=100)
    credential_payment_trap: int = Field(ge=0, le=100)
    synthetic_or_script_signals: int = Field(ge=0, le=100)


class ScamAnalysisResponse(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    verdict: Literal["Low Risk", "Suspicious", "High Risk"]
    scam_type: str
    confidence_level: Literal["Low", "Medium", "High"]
    signal_scores: SignalScores
    # Bound the returned payload in Pydantic without expanding Gemini's schema
    # into repeated maxItems branches, which can trigger INVALID_ARGUMENT/400.
    red_flags: list[str]
    explanation: str
    recommended_action: str
    mode: Literal["text", "audio"]
    offline_fallback_used: bool

    @field_validator("red_flags")
    @classmethod
    def cap_red_flags(cls, flags: list[str]) -> list[str]:
        if len(flags) > 20:
            raise ValueError("red_flags must contain no more than 20 items")
        return flags


SYSTEM_PROMPT = """You are ScamShield AI, a cautious scam and impersonation intelligence assistant.
Assess only evidence in the supplied message/transcript or audio. Do not claim to
authenticate a speaker, prove a voice is synthetic, or verify facts you cannot verify.
Score scam indicators, not a person's character. Be calibrated and non-alarmist.
Use Low Risk for little evidence, Suspicious for meaningful but inconclusive signals,
and High Risk only for strong scam indicators. Red flags must be concise and grounded
in the input. Give a practical, safe next step. For audio, describe only audible/content
evidence and state uncertainty. Return the required JSON schema."""

app = FastAPI(title="ScamShield AI", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def resolve_key(request_key: str | None) -> str:
    key = (request_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(400, "Enter a Gemini API key or set GEMINI_API_KEY in the server environment.")
    return key


def connectivity_failure(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    name, message = type(exc).__name__.lower(), str(exc).lower()
    return any(x in name for x in ("connectionerror", "connecterror", "timeout", "networkerror", "transporterror")) or any(
        x in message for x in ("connection refused", "name or service not known", "temporary failure in name resolution", "network is unreachable", "failed to connect", "connection reset", "timed out", "timeout")
    )


def provider_detail(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    if code in (401, 403):
        return "Gemini rejected the key or its API permissions. Use an AI Studio Gemini API key and confirm the Gemini API is enabled."
    if code == 404:
        return f"Gemini model '{MODEL}' was not found or is unavailable to this key. Check GEMINI_MODEL."
    if code == 429:
        return "Gemini quota or rate limit was reached. Check the project's Gemini API usage, quota, and billing."
    if code == 400:
        # Surface Google's concise validation reason so the developer can fix the
        # actual rejected field/schema. Keep this bounded and redact API-key shapes.
        raw = str(getattr(exc, "message", "") or str(exc)).replace("\r", " ").replace("\n", " ")
        raw = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted API key]", raw)
        raw = raw[:500].strip()
        return f"Gemini rejected the request (HTTP 400): {raw or 'invalid request/schema'}."
    if code in (500, 502, 503, 504):
        return f"Gemini is temporarily unavailable (HTTP {code}). Retry shortly."
    if code is not None:
        return f"Gemini returned HTTP {code}. Check the server log for details."
    return f"Gemini request failed ({type(exc).__name__}). Check the server log for details."


def local_fallback(text: str, mode: Literal["text", "audio"]) -> ScamAnalysisResponse:
    if mode == "audio":
        return ScamAnalysisResponse(
            risk_score=0, verdict="Low Risk", scam_type="Unassessed audio", confidence_level="Low",
            signal_scores=SignalScores(authority_impersonation=0, urgency_isolation=0, credential_payment_trap=0, synthetic_or_script_signals=0),
            red_flags=[], explanation="The audio could not be assessed because the analysis service is offline. No conclusion about this clip can be drawn locally.",
            recommended_action="Do not act on urgent requests in the clip. Verify the speaker through a trusted contact method and retry when online.",
            mode="audio", offline_fallback_used=True,
        )
    rules = {
        "authority_impersonation": (30, ("police", "court", "customs", "government", "official", "cbi", "rbi", "bank officer", "digital arrest")),
        "urgency_isolation": (25, ("urgent", "immediately", "today", "within 24 hours", "don't tell", "do not tell", "keep this secret", "arrest warrant")),
        "credential_payment_trap": (35, ("otp", "one-time password", "password", "pin", "upi", "transfer money", "send money", "bank details", "gift card", "crypto", "click this link")),
        "synthetic_or_script_signals": (10, ("voice clone", "ai generated", "synthetic voice", "deepfake")),
    }
    lower = text.casefold()
    scores: dict[str, int] = {}
    flags: list[str] = []
    for name, (weight, phrases) in rules.items():
        found = [phrase for phrase in phrases if phrase in lower]
        scores[name] = min(100, weight * len(found))
        flags.extend(f"Contains the phrase “{phrase}”" for phrase in found)
    score = min(100, sum(scores.values()))
    verdict: Literal["Low Risk", "Suspicious", "High Risk"] = "High Risk" if score >= 60 else "Suspicious" if score >= 25 else "Low Risk"
    return ScamAnalysisResponse(
        risk_score=score, verdict=verdict, scam_type="Potential social engineering" if flags else "No common scam phrase detected",
        confidence_level="Low", signal_scores=SignalScores(**scores), red_flags=flags[:20],
        explanation="Connectivity prevented full AI analysis. This low-confidence local keyword check found the listed phrases; they do not prove a scam." if flags else "Connectivity prevented full AI analysis. No configured scam phrases were found; that does not establish safety.",
        recommended_action="Do not share passwords, OTPs, or money based on this message. Verify independently using an official number or website.",
        mode="text", offline_fallback_used=True,
    )


def analyze(client: genai.Client, contents: list[object], mode: Literal["text", "audio"], fallback_text: str = "") -> ScamAnalysisResponse:
    try:
        result = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ScamAnalysisResponse,
            ),
        )
        parsed = result.parsed
        response = parsed if isinstance(parsed, ScamAnalysisResponse) else (
            ScamAnalysisResponse.model_validate(parsed) if parsed is not None
            else ScamAnalysisResponse.model_validate_json(result.text or "")
        )
        response.mode = mode
        response.offline_fallback_used = False
        return response
    except Exception as exc:
        if connectivity_failure(exc):
            logger.warning("Gemini connection failed; returning transparent offline fallback (%s)", type(exc).__name__)
            return local_fallback(fallback_text, mode)
        logger.exception("Gemini analysis failed; provider diagnostic follows")
        raise HTTPException(502, provider_detail(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def home() -> FileResponse:
    index = ROOT / "index.html"
    if not index.is_file():
        raise HTTPException(404, "Frontend file index.html was not found beside server.py.")
    return FileResponse(index, media_type="text/html")


@app.post("/api/analyze-text", response_model=ScamAnalysisResponse)
def analyze_text(text: Annotated[str, Form(...)], api_key: Annotated[str, Form()] = "") -> ScamAnalysisResponse:
    text = text.strip()
    if not text:
        raise HTTPException(400, "Text must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"Text exceeds the {MAX_TEXT_CHARS:,} character limit.")
    client = genai.Client(api_key=resolve_key(api_key))
    try:
        return analyze(client, [f"Analyze this suspicious message or call transcript:\n\n{text}"], "text", text)
    finally:
        client.close()


@app.post("/api/analyze-audio", response_model=ScamAnalysisResponse)
async def analyze_audio(file: Annotated[UploadFile, File(...)], api_key: Annotated[str, Form()] = "") -> ScamAnalysisResponse:
    suffix = Path(file.filename or "").suffix.lower()
    mime = (file.content_type or "").split(";", 1)[0].strip().lower()
    if suffix not in ALLOWED_EXTENSIONS or mime not in ALLOWED_MIME_TYPES:
        await file.close()
        raise HTTPException(415, "Unsupported audio format. Use MP3, WAV, M4A, OGG, or WEBM.")
    key = resolve_key(api_key)
    temp_path: str | None = None
    client: genai.Client | None = None
    remote_name: str | None = None
    try:
        size = 0
        with tempfile.NamedTemporaryFile(prefix="scamshield-", suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_AUDIO_BYTES:
                    raise HTTPException(413, "Audio file exceeds the 20 MB limit.")
                temp.write(chunk)
        if not size:
            raise HTTPException(400, "Audio file is empty.")
        client = genai.Client(api_key=key)
        uploaded = client.files.upload(file=temp_path)
        remote_name = getattr(uploaded, "name", None)
        return analyze(client, [uploaded, "Analyze this audio for scam, coercion, impersonation, and synthetic-voice indicators. State uncertainty."], "audio")
    except HTTPException:
        raise
    except Exception as exc:
        if connectivity_failure(exc):
            logger.warning("Audio analysis connectivity failure (%s)", type(exc).__name__)
            return local_fallback("", "audio")
        logger.exception("Audio upload or analysis failed; provider diagnostic follows")
        raise HTTPException(502, provider_detail(exc)) from exc
    finally:
        await file.close()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove local temporary audio file")
        if client is not None and remote_name:
            try:
                client.files.delete(name=remote_name)
            except Exception:
                logger.warning("Could not delete uploaded provider file %s", remote_name)
        if client is not None:
            client.close()


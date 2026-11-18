"""server.py — server de inferență ASR (română / rusă) + interfața web.

Pornire:
    pip install -r requirements.txt
    python server.py                      # sau: uvicorn server:app --port 8000
    apoi deschideți http://localhost:8000  (interfața web este servită de acest server)

Variabile de mediu:
    MODEL_DIR         modelul antrenat (implicit asr_project/model_final)
    WHISPER_MODEL     model Whisper pentru engine=whisper (implicit openai/whisper-small)
    ROBIN_API_URL     serverul Robin (implicit URL-ul din enunț; recomandat: http://localhost:8888/transcribe)
    RU_MODEL_ID       model rus HF (implicit bond005/wav2vec2-large-ru-golos)
    MAX_AUDIO_SEC     durata maximă acceptată (implicit 3600)
    FALLBACK_HF_MODEL model CTC HF folosit dacă nu există model antrenat local (opțional)

Endpoint-uri (Swagger: /docs):
    GET  /                       interfața web
    GET  /health                 starea serverului și a modelelor
    GET  /model-card             Data Card + rezumat antrenare + rezultate evaluare
    POST /transcribe             fișier audio/video (multipart: file, language, engine)
    POST /transcribe/base64      audio Base64 (JSON: audio, format, language, engine)
    POST /transcribe/batch       până la 10 fișiere
    GET/POST /api/validation...  validare manuală a setului de test (folosit de tab-ul „Validare")
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from asr import __version__ as VERSION
from asr.audio_io import AudioError, load_audio, write_wav
from asr.config import Config
from asr.inference import SR, Transcriber, is_model_dir, pick_device
from asr.labelers import HFLabeler, LabelerError, RobinEndpointError, RobinLabeler
from asr.segment import merge_spans, read_jsonl, speech_spans
from asr import validate as val

ROOT = pathlib.Path(__file__).resolve().parent
CFG = Config()
MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", CFG.final_dir))
ROBIN_URL = os.getenv("ROBIN_API_URL", CFG.robin_url)
RU_MODEL_ID = os.getenv("RU_MODEL_ID", CFG.ru_model)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
MAX_AUDIO_SEC = float(os.getenv("MAX_AUDIO_SEC", "3600"))
VAD = os.getenv("ASR_VAD", "silero")
DEVICE = pick_device(os.getenv("ASR_DEVICE"))

STATE: dict = {"whisper": None, "ro": None, "ro_error": None, "ru": None, "ru_error": None, "started": time.time()}
_RU_LOCK = threading.Lock()
_W_LOCK = threading.Lock()


def _load_ro():
    try:
        if is_model_dir(MODEL_DIR):
            STATE["ro"] = Transcriber(str(MODEL_DIR), device=DEVICE, vad=VAD)
        elif os.getenv("FALLBACK_HF_MODEL"):
            STATE["ro"] = Transcriber(os.environ["FALLBACK_HF_MODEL"], device=DEVICE, vad=VAD)
        else:
            STATE["ro_error"] = (f"Model local negăsit în {MODEL_DIR}. Antrenați-l (python run_pipeline.py train) "
                                 f"sau setați MODEL_DIR / FALLBACK_HF_MODEL. engine=robin rămâne disponibil.")
    except Exception as e:
        STATE["ro_error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_ro()
    print(f"✅ model RO: {'încărcat pe ' + DEVICE if STATE['ro'] else '— ' + str(STATE['ro_error'])}")
    yield


app = FastAPI(title="RO-EduASR — Server de inferență",
              description="Recunoaștere automată a vorbirii în română (model antrenat local sau RobinASR) și rusă.",
              version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ───────────────────────── modele de date ──────────────────────
class Base64Request(BaseModel):
    audio: str                 # Base64 (poate avea prefix data:...;base64,)
    format: str = "wav"        # extensia: wav, mp3, mp4, webm ...
    language: str = "ro"
    engine: str = "auto"


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    transcription: str
    segments: List[Segment] = []
    duration_audio: float
    processing_time: float
    rtf: float                 # Real-Time Factor (< 1 = mai rapid decât timpul real)
    language: str
    engine: str
    device: str
    model: str


# ───────────────────────── logică ──────────────────────────────
def _ru() -> Transcriber:
    with _RU_LOCK:               # încărcare o singură dată, nu la fiecare cerere
        if STATE["ru"] is None:
            try:
                STATE["ru"] = Transcriber(RU_MODEL_ID, device=DEVICE, vad=VAD)
            except Exception as e:
                STATE["ru_error"] = str(e)
                raise HTTPException(503, f"Modelul rus {RU_MODEL_ID} nu a putut fi încărcat "
                                         f"(necesită acces la huggingface.co): {e}")
    return STATE["ru"]


def _whisper() -> HFLabeler:
    with _W_LOCK:                # încărcare o singură dată (la prima cerere; descarcă modelul dacă lipsește)
        if STATE["whisper"] is None:
            try:
                STATE["whisper"] = HFLabeler(WHISPER_MODEL, CFG, device=DEVICE, batch_size=8)
            except Exception as e:
                raise HTTPException(503, f"Whisper ({WHISPER_MODEL}) nu a putut fi încărcat: {e}")
    return STATE["whisper"]


def _whisper_transcribe(audio: np.ndarray) -> dict:
    lab = _whisper()
    spans = merge_spans(speech_spans(audio, SR, VAD, 20.0), SR, 20.0, 0.6) or [(0, len(audio))]
    spans = [(s, e) for s, e in spans if e - s >= 0.3 * SR]
    texts = lab.transcribe_arrays([audio[s:e] for s, e in spans]) if spans else []
    segs = [{"start": round(s / SR, 2), "end": round(e / SR, 2), "text": t}
            for (s, e), t in zip(spans, texts) if t]
    return {"text": " ".join(x["text"] for x in segs), "segments": segs, "device": DEVICE, "model": WHISPER_MODEL}


def _robin_transcribe(audio: np.ndarray) -> dict:
    """Trimite audio la Robin pe segmente scurte (Robin e gândit pentru fragmente, nu lecții întregi)."""
    spans = merge_spans(speech_spans(audio, SR, VAD, 20.0), SR, 20.0, 0.6) or [(0, len(audio))]
    labeler = RobinLabeler(ROBIN_URL, timeout=CFG.robin_timeout)
    segs = []
    with tempfile.TemporaryDirectory() as td:
        for i, (s, e) in enumerate(spans):
            if e - s < 0.3 * SR:
                continue
            p = pathlib.Path(td) / f"seg{i}.wav"
            write_wav(p, audio[s:e], SR)
            txt = labeler.label({}, p)
            if txt:
                segs.append({"start": round(s / SR, 2), "end": round(e / SR, 2), "text": txt.lower()})
    return {"text": " ".join(x["text"] for x in segs), "segments": segs, "device": "remote", "model": ROBIN_URL}


def _run(audio: np.ndarray, language: str, engine: str) -> dict:
    t0 = time.perf_counter()
    dur = len(audio) / SR
    if dur > MAX_AUDIO_SEC:
        raise HTTPException(413, f"Audio prea lung ({dur:.0f}s). Maxim: {MAX_AUDIO_SEC:.0f}s (MAX_AUDIO_SEC).")
    if language not in ("ro", "ru"):
        raise HTTPException(400, "language trebuie să fie 'ro' sau 'ru'.")
    if engine not in ("auto", "local", "robin", "whisper"):
        raise HTTPException(400, "engine trebuie să fie 'auto', 'local', 'robin' sau 'whisper'.")
    try:
        if language == "ru":
            res, used = _ru().transcribe(audio), "local"
        else:
            use = engine if engine != "auto" else ("local" if STATE["ro"] else "robin")
            if use == "local":
                if not STATE["ro"]:
                    raise HTTPException(503, STATE["ro_error"] or "Modelul local nu este încărcat.")
                res, used = STATE["ro"].transcribe(audio), "local"
            elif use == "whisper":
                res, used = _whisper_transcribe(audio), "whisper"
            else:
                res, used = _robin_transcribe(audio), "robin"
    except RobinEndpointError as e:
        extra = ("\n\nAlternativ: alegeți motorul „Whisper” (fără Docker, se descarcă la prima utilizare) "
                 "sau antrenați modelul local.") if engine == "auto" else ""
        raise HTTPException(502, str(e) + extra)
    except LabelerError as e:
        raise HTTPException(502, f"Robin: {e}")
    elapsed = time.perf_counter() - t0
    return {"transcription": res["text"], "segments": res.get("segments", []),
            "duration_audio": round(dur, 3), "processing_time": round(elapsed, 3),
            "rtf": round(elapsed / max(dur, 1e-3), 4), "language": language, "engine": used,
            "device": res.get("device", DEVICE), "model": res.get("model", "")}


def _decode_bytes(data: bytes, suffix: str) -> np.ndarray:
    suffix = suffix if suffix.startswith(".") else "." + suffix
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return load_audio(path, SR)
    except AudioError as e:
        raise HTTPException(400, str(e))
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


# ───────────────────────── endpoint-uri ────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "asr_interface.html")


@app.get("/health", tags=["Status"])
def health():
    import shutil
    return {"status": "ok", "version": VERSION, "device": DEVICE,
            "model_ro_loaded": STATE["ro"] is not None, "model_ro_dir": str(MODEL_DIR),
            "model_ro_error": STATE["ro_error"], "model_ru_id": RU_MODEL_ID,
            "model_ru_loaded": STATE["ru"] is not None, "whisper_model": WHISPER_MODEL,
            "robin_url": ROBIN_URL, "ffmpeg": shutil.which("ffmpeg") is not None,
            "uptime_s": round(time.time() - STATE["started"])}


@app.get("/model-card", tags=["Status"])
def model_card():
    out = {}
    for key, p in (("data_card", CFG.data_card_json),
                   ("training", MODEL_DIR / "training_summary.json"),
                   ("evaluation_test", CFG.results_dir / "eval_test.json")):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if key == "training":
                d.pop("log_history", None)
            out[key] = d
    if not out:
        return JSONResponse(status_code=404, content={
            "detail": "Încă nu există Data Card / model. Rulați: python run_pipeline.py datacard"})
    return out


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["ASR"])
def transcribe_file(file: UploadFile = File(..., description="Audio sau video: WAV, MP3, MP4, WebM, MKV ..."),
                    language: str = Form("ro"), engine: str = Form("auto")):
    """Transcriere dintr-un fișier audio/video. Câmpuri multipart: `file`, `language` (ro|ru),
    `engine` (auto|local|robin)."""
    audio = _decode_bytes(file.file.read(), pathlib.Path(file.filename or "audio.wav").suffix or ".wav")
    return _run(audio, language, engine)


@app.post("/transcribe/base64", response_model=TranscriptionResponse, tags=["ASR"])
def transcribe_base64(body: Base64Request):
    raw = body.audio.split(",", 1)[1] if body.audio.startswith("data:") else body.audio
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as e:
        raise HTTPException(400, f"Base64 invalid: {e}")
    return _run(_decode_bytes(data, "." + body.format.lstrip(".")), body.language, body.engine)


@app.post("/transcribe/batch", tags=["ASR"])
def transcribe_batch(files: List[UploadFile] = File(...), language: str = Form("ro"), engine: str = Form("auto")):
    if len(files) > 10:
        raise HTTPException(400, "Maxim 10 fișiere per cerere.")
    results = []
    for f in files:
        try:
            audio = _decode_bytes(f.file.read(), pathlib.Path(f.filename or "a.wav").suffix or ".wav")
            r = _run(audio, language, engine)
            results.append({"filename": f.filename, "status": "ok", **r})
        except HTTPException as e:
            results.append({"filename": f.filename, "status": "error", "error": e.detail, "transcription": ""})
    return {"results": results, "total": len(results), "language": language}


# ──────────────── validare manuală (tab „Validare") ────────────
class Decision(BaseModel):
    id: str
    status: str                 # confirmed | edited | deleted
    text: Optional[str] = ""


def _segments_index() -> dict:
    idx = {}
    for name in ("segments", "train", "dev", "test"):
        for s in read_jsonl(CFG.manifests_dir / f"{name}.jsonl"):
            idx[s["id"]] = s
    return idx


@app.get("/api/manifest/{split}", tags=["Validare"])
def api_manifest(split: str, selected: bool = False):
    if split not in ("train", "dev", "test"):
        raise HTTPException(404, "split necunoscut")
    rows = read_jsonl(CFG.manifests_dir / f"{split}.jsonl")
    if selected:
        rows = [r for r in rows if r.get("val_selected")]
    dec = val.load_decisions(CFG)
    for r in rows:
        r["decision"] = dec.get(r["id"])
    return rows


@app.get("/api/segment_audio/{seg_id}", tags=["Validare"])
def api_segment_audio(seg_id: str):
    seg = _segments_index().get(seg_id)
    if not seg:
        raise HTTPException(404, "segment necunoscut")
    p = CFG.resolve(seg["audio_filepath"]).resolve()
    if CFG.segments_dir.resolve() not in p.parents or not p.exists():     # fără path traversal
        raise HTTPException(404, "fișier indisponibil")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/validation", tags=["Validare"])
def api_validation():
    return list(val.load_decisions(CFG).values())


@app.post("/api/validation", tags=["Validare"])
def api_validation_post(d: Decision):
    if d.id not in _segments_index():
        raise HTTPException(404, "segment necunoscut")
    try:
        return val.record_decision(CFG, d.id, d.text, d.status)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/validation/apply", tags=["Validare"])
def api_validation_apply():
    return val.apply_validation(CFG)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))

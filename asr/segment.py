"""Pasul 3 (preprocesare) — extragere audio + segmentare VAD.

Audio-ul fiecărei înregistrări este împărțit în segmente de vorbire de 1–20 s.
Aceeași funcție `speech_spans` este folosită și de serverul de inferență, astfel
încât modelul vede la inferență același tip de segmente ca la antrenare.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Sequence

import numpy as np

from .audio_io import dbfs, extract_audio, load_audio, write_wav
from .collect import load_inventory
from .config import Config

_SILERO = {"model": None}


def _silero_model():
    if _SILERO["model"] is None:
        from silero_vad import load_silero_vad   # modelul vine în pachetul pip (fără descărcare)
        _SILERO["model"] = load_silero_vad()
    return _SILERO["model"]


# ───────────────────────── VAD ─────────────────────────────────
def _energy_spans(audio: np.ndarray, sr: int, max_sec: float) -> list[tuple[int, int]]:
    """VAD simplu pe energie (rezervă dacă silero lipsește; folosit și în teste)."""
    fl = int(0.03 * sr)
    n = len(audio) // fl
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * fl].reshape(n, fl) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = max(float(np.percentile(db, 95)) - 30.0, -60.0)
    active = db > thr
    # unește pauzele < 300 ms (10 cadre) și elimină exploziile < 150 ms (5 cadre)
    frames, start, last, gap = [], None, 0, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > 10:
                frames.append((start, last + 1)); start = None
    if start is not None:
        frames.append((start, last + 1))
    out = [(s * fl, e * fl) for s, e in frames if e - s >= 5]
    return _split_long(out, audio, sr, max_sec)


def _split_long(spans, audio, sr, max_sec):
    """Taie spanurile > max_sec în cel mai liniștit punct din ultimele 30% ale ferestrei."""
    mx = int(max_sec * sr)
    res = []
    for s, e in spans:
        while e - s > mx:
            lo, hi = s + int(0.7 * mx), s + mx
            win = 480
            seg = audio[lo:hi]
            nwin = max(len(seg) // win, 1)
            r = np.sqrt(np.mean(seg[: nwin * win].reshape(nwin, win) ** 2, axis=1))
            cut = lo + int(np.argmin(r)) * win + win // 2
            res.append((s, cut)); s = cut
        res.append((s, e))
    return res


def speech_spans(audio: np.ndarray, sr: int = 16_000, vad: str = "silero",
                 max_sec: float = 20.0) -> list[tuple[int, int]]:
    """Intervale (eșantion_start, eșantion_sfârșit) cu vorbire, fiecare ≤ max_sec."""
    if vad == "silero":
        try:
            import torch
            from silero_vad import get_speech_timestamps
            ts = get_speech_timestamps(
                torch.from_numpy(audio), _silero_model(), sampling_rate=sr,
                min_speech_duration_ms=250, min_silence_duration_ms=300,
                speech_pad_ms=100, max_speech_duration_s=max_sec)
            return [(int(t["start"]), int(t["end"])) for t in ts]
        except ImportError:
            print("⚠ silero-vad indisponibil — folosesc VAD pe energie")
    return _energy_spans(audio, sr, max_sec)


def merge_spans(spans: Sequence[tuple[int, int]], sr: int, max_sec: float, max_gap_sec: float):
    """Unește segmente vecine (pauză < max_gap_sec) cât timp rezultatul ≤ max_sec.
    Segmentele prea scurte (1–3 s) au puțin context și transcrieri slabe."""
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) <= max_gap_sec * sr and (e - merged[-1][0]) <= max_sec * sr:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ───────────────────── pipeline pe fișiere ─────────────────────
def extract_all(cfg: Config, inventory: list[dict] | None = None, loudnorm: bool = False) -> list[dict]:
    """Pasul 3a: video/audio -> WAV mono 16 kHz în audio_raw/ (sare peste cele existente)."""
    cfg.make_dirs()
    inventory = inventory if inventory is not None else load_inventory(cfg)
    done, errors = 0, []
    for it in inventory:
        dst = cfg.audio_dir / f"{it['recording_id']}.wav"
        try:
            extract_audio(cfg.resolve(it["file"]), dst, cfg.sampling_rate, loudnorm=loudnorm)
            it["wav"] = cfg.rel(dst); done += 1
        except Exception as e:
            errors.append((it["recording_id"], str(e))); print(f"  ⚠ {it['recording_id']}: {e}")
    print(f"✅ Audio extras: {done}/{len(inventory)}")
    return inventory


def segment_recording(cfg: Config, wav_path, rec: dict) -> list[dict]:
    audio = load_audio(wav_path, cfg.sampling_rate)
    sr = cfg.sampling_rate
    spans = speech_spans(audio, sr, cfg.vad, cfg.seg_max_sec)
    spans = merge_spans(spans, sr, cfg.seg_max_sec, cfg.seg_merge_gap_sec)
    out = []
    for i, (s, e) in enumerate(spans):
        chunk = audio[s:e]
        dur = len(chunk) / sr
        if not (cfg.seg_min_sec <= dur <= cfg.seg_max_sec + 0.5):
            continue
        if dbfs(chunk) < cfg.seg_min_dbfs:
            continue
        seg_id = f"{rec['recording_id']}_{i:04d}"
        path = cfg.segments_dir / rec["recording_id"] / f"{seg_id}.wav"
        write_wav(path, chunk, sr)
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        out.append({
            "id": seg_id,
            "recording_id": rec["recording_id"],
            "speaker": rec.get("speaker") or rec["recording_id"],
            "source": rec["source"],
            "audio_filepath": cfg.rel(path),
            "start": round(s / sr, 3), "end": round(e / sr, 3), "duration": round(dur, 3),
            "md5": hashlib.md5(pcm).hexdigest(),
        })
    return out


def segment_all(cfg: Config, inventory: list[dict] | None = None) -> list[dict]:
    """Pasul 3b: VAD pe fiecare WAV, deduplicare md5, scriere manifests/segments.jsonl."""
    cfg.make_dirs()
    inventory = inventory if inventory is not None else load_inventory(cfg)
    all_segs, seen = [], set()
    for rec in inventory:
        wav = cfg.audio_dir / f"{rec['recording_id']}.wav"
        if not wav.exists():
            print(f"  ⚠ lipsește {wav.name} (rulați extract)"); continue
        segs = segment_recording(cfg, wav, rec)
        kept = [s for s in segs if s["md5"] not in seen]
        seen.update(s["md5"] for s in kept)
        all_segs.extend(kept)
        print(f"  {rec['recording_id']}: {len(kept)} segmente")
    write_jsonl(cfg.manifests_dir / "segments.jsonl", all_segs)
    hrs = sum(s["duration"] for s in all_segs) / 3600
    print(f"✅ {len(all_segs)} segmente unice, {hrs:.2f} ore de vorbire → manifests/segments.jsonl")
    return all_segs


# ─────────────────────── utilitare manifest ────────────────────
def write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

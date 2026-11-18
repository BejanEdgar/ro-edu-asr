"""Date sintetice pentru testul rapid („smoke test") al întregului pipeline.

NU este vorbire reală. Fiecare literă este un ton cu frecvență proprie; textul este deci un „cod audio" pe
care un model minuscul îl poate învăța pe CPU în câteva minute. Scopul este să dovedească faptul că
extragerea audio, VAD-ul, etichetarea, split-ul, antrenarea, evaluarea și serverul funcționează
împreună — fără rețea, fără GPU și fără datele reale (care nu pot fi descărcate în orice mediu).
"""
from __future__ import annotations

import json
import pathlib
import random
import subprocess

import numpy as np

from .audio_io import write_wav
from .config import Config
from .text_norm import VOCAB_CHARS

WORDS = ("adunarea scăderea înmulțirea împărțirea numerele fracțiile perimetrul pătratul dreptunghiul "
         "unitățile măsura lungimea masa timpul calendarul problema exercițiul rezultatul suma diferența "
         "produsul câtul numitorul ceasul banii kilogramul metrul litrul ordinul zecile sutele miile "
         "doi plus trei egal cinci").split()
SR = 16_000
TONE, GAP, SPACE = 0.070, 0.020, 0.040


def _freq(ch: str) -> float:
    return 250.0 + 58.0 * VOCAB_CHARS.index(ch)


def synth_text_audio(text: str, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for ch in text:
        if ch == " ":
            parts.append(np.zeros(int(SPACE * SR), np.float32)); continue
        n = int(TONE * SR)
        t = np.arange(n) / SR
        w = 0.35 * np.sin(2 * np.pi * _freq(ch) * t).astype(np.float32)
        fade = int(0.004 * SR)
        w[:fade] *= np.linspace(0, 1, fade); w[-fade:] *= np.linspace(1, 0, fade)
        parts.append(w); parts.append(np.zeros(int(GAP * SR), np.float32))
    a = np.concatenate(parts)
    return a + rng.normal(0, 0.002, a.shape).astype(np.float32)


def make_corpus(cfg: Config, n_rec: int = 10, utt_per_rec: int = 18, seed: int = 0,
                as_video_every: int = 3) -> dict:
    """Creează înregistrări în videos/manual/. Returnează adevărul de teren (pentru oracol/teste)."""
    cfg.make_dirs()
    rng, py = np.random.default_rng(seed), random.Random(seed)
    out_dir = cfg.videos_dir / "manual"
    truth: dict[str, list[dict]] = {}
    for r in range(n_rec):
        pieces, t, items = [np.zeros(int(1.0 * SR), np.float32)], 1.0, []
        for _ in range(utt_per_rec):
            text = " ".join(py.choice(WORDS) for _ in range(py.randint(2, 3)))
            a = synth_text_audio(text, rng)
            items.append({"start": t, "end": t + len(a) / SR, "text": text})
            pause = np.zeros(int(py.uniform(1.2, 1.8) * SR), np.float32)
            pieces += [a, pause]; t += (len(a) + len(pause)) / SR
        audio = np.concatenate(pieces)
        name = f"synth_{r:02d}"
        wav = out_dir / f"{name}.wav"
        write_wav(wav, audio, SR)
        if as_video_every and r % as_video_every == 0:          # testează și calea video -> audio
            mp4 = out_dir / f"{name}.mp4"
            dur = len(audio) / SR
            cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                   f"color=c=black:s=64x64:r=5:d={dur:.2f}", "-i", str(wav), "-shortest",
                   "-c:v", "mpeg4", "-c:a", "aac", str(mp4)]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                wav.unlink()
        truth[f"SRC03_{name}"] = items
    (cfg.manifests_dir / "synth_truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
    return truth


def oracle_text(truth: dict, seg: dict) -> str:
    """Textul real al segmentului = enunțurile acoperite în cea mai mare parte de segment."""
    words = []
    for u in truth.get(seg["recording_id"], []):
        ov = min(u["end"], seg["end"]) - max(u["start"], seg["start"])
        if ov > 0.5 * (u["end"] - u["start"]):
            words.append(u["text"])
    return " ".join(words)

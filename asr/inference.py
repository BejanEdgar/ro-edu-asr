"""Inferență ASR cu model Wav2Vec2-CTC (folosită de evaluare, de etichetare și de server).

Audio lung (lecții de zeci de minute) este împărțit cu același VAD ca la antrenare, în
segmente ≤ 20 s transcrise în batch-uri, cu timestamp-uri per segment.
"""
from __future__ import annotations

import pathlib
import time
from typing import List, Optional

import numpy as np
import torch

from .segment import merge_spans, speech_spans

SR = 16_000


def pick_device(device: Optional[str] = None) -> str:
    if device in ("cpu", "cuda"):
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class Transcriber:
    def __init__(self, model_dir_or_id: str, device: Optional[str] = None, vad: str = "silero",
                 batch_size: int = 8, max_sec: float = 20.0):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.name = str(model_dir_or_id)
        self.device = pick_device(device)
        self.vad = vad               # 'silero' | 'energy' | 'none'
        self.batch_size = batch_size
        self.max_sec = max_sec
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir_or_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir_or_id).to(self.device).eval()
        self.use_mask = bool(getattr(self.processor.feature_extractor, "return_attention_mask", False))

    @torch.inference_mode()
    def transcribe_batch(self, arrays: List[np.ndarray]) -> List[str]:
        """Transcrie o listă de semnale (≤ ~20 s fiecare). Păstrează ordinea."""
        order = sorted(range(len(arrays)), key=lambda i: len(arrays[i]))
        out = [""] * len(arrays)
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            chunk = [arrays[i] for i in idx]
            feats = self.processor.feature_extractor(
                chunk, sampling_rate=SR, padding=True, return_tensors="pt",
                return_attention_mask=self.use_mask)
            kw = {"attention_mask": feats["attention_mask"].to(self.device)} if self.use_mask else {}
            use_amp = self.device == "cuda"
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = self.model(feats["input_values"].to(self.device), **kw).logits
            texts = self.processor.batch_decode(torch.argmax(logits, dim=-1))
            for i, t in zip(idx, texts):
                out[i] = t.strip()
        return out

    def plan(self, audio: np.ndarray) -> list[tuple[int, int]]:
        spans = speech_spans(audio, SR, self.vad, self.max_sec)
        spans = merge_spans(spans, SR, self.max_sec, 0.6)
        if not spans and len(audio) <= self.max_sec * SR:
            spans = [(0, len(audio))]          # înregistrare scurtă pe care VAD n-a prins-o
        return [(s, e) for s, e in spans if e - s >= 0.2 * SR]

    def transcribe(self, audio: np.ndarray, with_segments: bool = True) -> dict:
        t0 = time.perf_counter()
        dur = len(audio) / SR
        if self.vad == "none":   # fără VAD: tăiere fixă dacă e prea lung
            n = int(self.max_sec * SR)
            spans = [(i, min(i + n, len(audio))) for i in range(0, len(audio), n)]
        else:
            spans = self.plan(audio)
        texts = self.transcribe_batch([audio[s:e] for s, e in spans]) if spans else []
        segs = [{"start": round(s / SR, 2), "end": round(e / SR, 2), "text": t}
                for (s, e), t in zip(spans, texts) if t]
        elapsed = time.perf_counter() - t0
        res = {"text": " ".join(x["text"] for x in segs).strip(), "duration": round(dur, 3),
               "processing_time": round(elapsed, 3), "rtf": round(elapsed / max(dur, 1e-3), 4),
               "device": self.device, "model": self.name}
        if with_segments:
            res["segments"] = segs
        return res


def is_model_dir(path) -> bool:
    p = pathlib.Path(path)
    return p.is_dir() and (p / "config.json").exists() and (p / "vocab.json").exists()

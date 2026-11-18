"""Pasul 5 — vocabular, procesor, Dataset PyTorch și collator CTC.

Datasetul citește fișierele WAV „la cerere" (nu le ține pe toate în RAM) și nu depinde de
`datasets.Audio` (care în versiunile noi cere torchcodec și schimbă formatul `['array']`)
sau de `processor.as_target_processor()` (eliminat în transformers 5).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import Config
from .segment import read_jsonl
from .text_norm import VOCAB_CHARS


def build_vocab() -> dict:
    """Convenția HF pentru CTC: spațiul devine '|' (delimitator de cuvinte)."""
    chars = ["|" if c == " " else c for c in VOCAB_CHARS]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def make_processor(save_dir=None, sampling_rate: int = 16_000, return_attention_mask: bool = True):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    save_dir = pathlib.Path(save_dir) if save_dir else None
    vocab = build_vocab()
    tmp = (save_dir or pathlib.Path(".")) / "vocab.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(tmp), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sampling_rate, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=return_attention_mask)
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
    if save_dir:
        proc.save_pretrained(str(save_dir))
    return proc


class ASRDataset(Dataset):
    def __init__(self, segments: List[dict], processor, cfg: Config, use_text: bool = True,
                 max_sec: float | None = None):
        self.cfg = cfg
        self.processor = processor
        self.use_text = use_text
        keep = [s for s in segments if not s.get("excluded") and (s.get("text") or not use_text)]
        if max_sec:
            keep = [s for s in keep if s["duration"] <= max_sec]
        self.segments = keep
        self.lengths = [int(s["duration"] * cfg.sampling_rate) for s in keep]

    def __len__(self):
        return len(self.segments)

    def _audio(self, seg) -> np.ndarray:
        audio, sr = sf.read(str(self.cfg.resolve(seg["audio_filepath"])), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.cfg.sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sampling_rate)
        return audio

    def __getitem__(self, i):
        seg = self.segments[i]
        feats = self.processor.feature_extractor(
            self._audio(seg), sampling_rate=self.cfg.sampling_rate).input_values[0]
        item = {"input_values": np.asarray(feats, dtype=np.float32), "id": seg["id"]}
        if self.use_text:
            item["labels"] = self.processor.tokenizer(seg["text"]).input_ids
        return item


@dataclass
class DataCollatorCTC:
    pad_token_id: int
    return_attention_mask: bool = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(features)
        L = max(len(f["input_values"]) for f in features)
        x = torch.zeros(B, L, dtype=torch.float32)
        mask = torch.zeros(B, L, dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_values"])
            x[i, :n] = torch.from_numpy(np.asarray(f["input_values"], dtype=np.float32))
            mask[i, :n] = 1
        batch = {"input_values": x}
        if self.return_attention_mask:
            batch["attention_mask"] = mask
        if "labels" in features[0]:
            T = max(len(f["labels"]) for f in features)
            lab = torch.full((B, T), -100, dtype=torch.long)
            for i, f in enumerate(features):
                lab[i, :len(f["labels"])] = torch.tensor(f["labels"], dtype=torch.long)
            batch["labels"] = lab
        return batch


def load_split(cfg: Config, name: str) -> List[dict]:
    rows = read_jsonl(cfg.manifests_dir / f"{name}.jsonl")
    if not rows:
        raise FileNotFoundError(f"manifests/{name}.jsonl lipsește sau e gol — rulați pasul 'split'.")
    return rows

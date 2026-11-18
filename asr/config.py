"""Configurare centrală. Toate căile sunt relative la `asr_project/`.

Valorile pot fi suprascrise prin variabile de mediu (vezi README) sau prin
`Config(base_dir=...)` (folosit de testele automate).
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Id-ul folderului Google Drive furnizat în enunțul lucrării
GDRIVE_FOLDER_ID = "1-7Zvo4p4ilABJVmT5izZrkpyT1P6Slvm"
SOURCE_URL_RO = "https://educatieonline.md/Video?class=4&discipline=6"
ROBIN_DEFAULT_URL = "https://relate.racai.ro/index.php?path=robin/asr"

# Sursele de date (id -> metadate). Folosit de colectare și de Data Card.
SOURCES = {
    "SRC-01": {
        "name": "EducațieOnline.md — clasa a 4-a, Matematică (lecții video, YouTube)",
        "url": SOURCE_URL_RO,
        "subdir": "educatieonline",
        "prefix": "SRC01",
        "language": "ro",
    },
    "SRC-02": {
        "name": "Google Drive — materiale video furnizate în enunțul lucrării",
        "url": f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
        "subdir": "gdrive",
        "prefix": "SRC02",
        "language": "ro",
    },
    "SRC-03": {
        "name": "Fișiere adăugate manual (audio/video)",
        "url": "",
        "subdir": "manual",
        "prefix": "SRC03",
        "language": "ro",
    },
}

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".m4v", ".flv"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac"}


def _env(name: str, default):
    v = os.getenv(name)
    return default if v in (None, "") else type(default)(v)


@dataclass
class Config:
    base_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(os.getenv("ASR_PROJECT_DIR", ROOT / "asr_project"))
    )

    # ── audio ──────────────────────────────────────────────────
    sampling_rate: int = 16_000
    seed: int = 42

    # ── segmentare ─────────────────────────────────────────────
    vad: str = "silero"            # 'silero' | 'energy'
    seg_min_sec: float = 1.0
    seg_max_sec: float = 20.0
    seg_merge_gap_sec: float = 0.6
    seg_min_dbfs: float = -55.0    # segmente mai silențioase se elimină

    # ── transcrieri ────────────────────────────────────────────
    robin_url: str = field(default_factory=lambda: os.getenv("ROBIN_API_URL", ROBIN_DEFAULT_URL))
    robin_timeout: int = 60
    label_workers: int = 4
    min_chars_per_sec: float = 3.0    # sub / peste -> transcriere suspectă
    max_chars_per_sec: float = 30.0

    # ── împărțirea datelor ─────────────────────────────────────
    split_train: float = 0.8
    split_dev: float = 0.1
    split_test: float = 0.1
    validation_fraction: float = 0.10   # din setul de test
    validation_min_segments: int = 30

    # ── model ──────────────────────────────────────────────────
    base_model: str = field(
        default_factory=lambda: os.getenv("BASE_MODEL", "facebook/wav2vec2-large-xlsr-53")
    )
    ru_model: str = field(
        default_factory=lambda: os.getenv("RU_MODEL_ID", "bond005/wav2vec2-large-ru-golos")
    )

    # ── antrenare ──────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 5
    gradient_checkpointing: bool = True
    freeze_feature_encoder: bool = True

    # ── căi derivate ───────────────────────────────────────────
    def __post_init__(self):
        self.base_dir = pathlib.Path(self.base_dir)

    @property
    def videos_dir(self): return self.base_dir / "videos"
    @property
    def audio_dir(self): return self.base_dir / "audio_raw"
    @property
    def segments_dir(self): return self.base_dir / "segments"
    @property
    def manifests_dir(self): return self.base_dir / "manifests"
    @property
    def checkpoints_dir(self): return self.base_dir / "model_checkpoints"
    @property
    def final_dir(self): return self.base_dir / "model_final"
    @property
    def results_dir(self): return self.base_dir / "results"
    @property
    def data_card_json(self): return self.base_dir / "data_card.json"
    @property
    def data_card_md(self): return self.base_dir / "DATA_CARD.md"

    def make_dirs(self) -> "Config":
        for d in (self.videos_dir, self.audio_dir, self.segments_dir, self.manifests_dir,
                  self.checkpoints_dir, self.final_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)
        for s in SOURCES.values():
            (self.videos_dir / s["subdir"]).mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, p) -> str:
        """Cale relativă la base_dir (POSIX) — manifestele rămân portabile între mașini."""
        p = pathlib.Path(p)
        try:
            return p.resolve().relative_to(self.base_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def resolve(self, p) -> pathlib.Path:
        p = pathlib.Path(p)
        return p if p.is_absolute() else self.base_dir / p

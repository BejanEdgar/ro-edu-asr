"""Pasul 4 — set de antrenare / dezvoltare / test.

Împărțirea se face pe GRUPURI (vorbitor = profesor, altfel înregistrare), nu pe segmente.
Segmentele vecine din aceeași lecție (aceeași voce, același subiect, aceleași cuvinte) sunt
aproape duplicate: dacă ar fi amestecate între train și test, WER-ul ar fi optimist.
Aceasta era o scurgere de date în împărțirea originală (pe segmente).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from .config import Config
from .segment import read_jsonl, write_jsonl


def group_split(segments: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 42,
                key: str = "speaker") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        groups[s.get(key) or s["recording_id"]].append(s)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(s["duration"] for s in segments) or 1.0
    t_test, t_dev = ratios[2] * total, ratios[1] * total

    out = {"train": [], "dev": [], "test": []}
    if len(names) >= 3:
        dur = lambda g: sum(s["duration"] for s in groups[g])
        got_test = got_dev = 0.0
        for g in names:
            if got_test < t_test:
                out["test"] += groups[g]; got_test += dur(g)
            elif got_dev < t_dev:
                out["dev"] += groups[g]; got_dev += dur(g)
            else:
                out["train"] += groups[g]
        if not out["train"]:   # cazuri extreme: puține grupuri foarte mari
            out["train"], out["dev"] = out["dev"], []
    else:
        # <3 grupuri: împărțire cronologică în blocuri contigue în fiecare înregistrare
        print("⚠ Sub 3 vorbitori/înregistrări — împărțire cronologică (risc mic de scurgere la graniță).")
        for g in names:
            segs = sorted(groups[g], key=lambda s: s["start"])
            tot = sum(s["duration"] for s in segs) or 1.0
            acc = 0.0
            for s in segs:
                frac = acc / tot
                part = "train" if frac < ratios[0] else ("dev" if frac < ratios[0] + ratios[1] else "test")
                out[part].append(s); acc += s["duration"]
    return out


def select_validation(test: list[dict], fraction: float, min_segments: int, seed: int = 42) -> list[dict]:
    """Alege ~fraction din test pentru validare manuală, distribuit pe înregistrări."""
    n = min(len(test), max(min_segments, math.ceil(fraction * len(test))))
    by_rec: dict[str, list[dict]] = defaultdict(list)
    for s in test:
        by_rec[s["recording_id"]].append(s)
    rng = random.Random(seed)
    for v in by_rec.values():
        rng.shuffle(v)
    picked, order = [], sorted(by_rec)
    rng.shuffle(order)
    while len(picked) < n and any(by_rec.values()):
        for r in order:
            if by_rec[r] and len(picked) < n:
                picked.append(by_rec[r].pop())
    ids = {s["id"] for s in picked}
    for s in test:
        s["val_selected"] = s["id"] in ids
    return picked


def make_splits(cfg: Config, segments: list[dict] | None = None) -> dict[str, list[dict]]:
    segments = segments if segments is not None else read_jsonl(cfg.manifests_dir / "segments.jsonl")
    usable = [s for s in segments if s.get("usable")]
    if not usable:
        raise RuntimeError("Nu există segmente etichetate utilizabile (rulați pasul de etichetare).")
    splits = group_split(usable, (cfg.split_train, cfg.split_dev, cfg.split_test), cfg.seed)
    picked = select_validation(splits["test"], cfg.validation_fraction, cfg.validation_min_segments, cfg.seed)
    for name, rows in splits.items():
        write_jsonl(cfg.manifests_dir / f"{name}.jsonl", rows)
        hrs = sum(s["duration"] for s in rows) / 3600
        spk = len({s["speaker"] for s in rows})
        print(f"  {name:5s}: {len(rows):5d} segmente | {hrs:6.2f} h | {spk} vorbitori/grupuri")
    print(f"✅ Split salvat în manifests/. De validat manual: {len(picked)} segmente din test "
          f"({len(picked) / max(len(splits['test']), 1):.0%}).")
    return splits

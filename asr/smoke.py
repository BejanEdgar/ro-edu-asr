"""Test end-to-end pe date sintetice (fără rețea/GPU): colectare -> extragere -> VAD -> etichetare ->
split -> validare -> Data Card -> antrenare -> evaluare."""
from __future__ import annotations

import json
import pathlib

from . import collect, datacard, segment, splits, validate
from .config import Config
from .evaluate import evaluate_split
from .labelers import MappingLabeler, label_segments
from .synth import make_corpus, oracle_text
from .train import train


def run_smoke(workdir, max_steps: int = 600, n_rec: int = 10, utt_per_rec: int = 18, train_model: bool = True) -> dict:
    cfg = Config(base_dir=pathlib.Path(workdir), vad="energy", label_workers=2,
                 validation_min_segments=5, validation_fraction=0.5).make_dirs()
    print("▶ 1/9 date sintetice"); truth = make_corpus(cfg, n_rec, utt_per_rec)
    print("▶ 2/9 inventar"); inv = collect.build_inventory(cfg); collect.save_inventory(cfg, inv)
    print("▶ 3/9 extragere audio"); inv = segment.extract_all(cfg, inv)
    print("▶ 4/9 segmentare VAD"); segs = segment.segment_all(cfg, inv)
    print("▶ 5/9 etichetare (oracol în loc de Robin)")
    lab = MappingLabeler(lambda s: oracle_text(truth, s), name="oracle-smoke")
    label_segments(cfg, segs, lab, save_path=cfg.manifests_dir / "segments.jsonl", verbose=False)
    print("▶ 6/9 split"); sp = splits.make_splits(cfg, segs)
    print("▶ 7/9 validare (om simulat = oracol)")
    for s in sp["test"]:
        if s.get("val_selected"):
            validate.record_decision(cfg, s["id"], oracle_text(truth, s), "confirmed")
    validate.apply_validation(cfg)
    print("▶ 8/9 Data Card"); datacard.write_data_card(cfg)
    res = {"workdir": str(workdir), "segments": len(segs),
           "split": {k: len(v) for k, v in sp.items()}}
    if train_model:
        print("▶ 9/9 antrenare + evaluare")
        res["train"] = {k: v for k, v in train(cfg, smoke=True, max_steps=max_steps).items() if k != "log_history"}
        res["eval"] = evaluate_split(cfg, "test")
        datacard.write_data_card(cfg)
    return res

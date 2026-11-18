#!/usr/bin/env python3
"""RO-EduASR — rulează pașii lucrării din linia de comandă.

    python run_pipeline.py collect                 # 1. colectare date (educatieonline.md + Google Drive)
    python run_pipeline.py extract                 # 3a. video -> WAV 16 kHz
    python run_pipeline.py segment                 # 3b. VAD -> segmente
    python run_pipeline.py label --labeler robin   # 3c. transcrieri (Robin / hf / csv)
    python run_pipeline.py split                   # 4.  train / dev / test (pe vorbitor)
    python run_pipeline.py validate-export         # 4.  CSV pentru validare manuală (sau tab-ul „Validare")
    python run_pipeline.py validate-apply
    python run_pipeline.py datacard                # 2.  Data Card (JSON + Markdown)
    python run_pipeline.py train                   # 5–6. antrenare
    python run_pipeline.py evaluate                # 7.  evaluare
    python run_pipeline.py serve                   # server + interfață web
    python run_pipeline.py prepare                 # extract + segment + label + split + datacard
    python run_pipeline.py smoke                   # test complet pe date sintetice (fără rețea/GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

from asr.config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", help="director de lucru (implicit asr_project/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect"); c.add_argument("--sources", nargs="+", default=["SRC-01", "SRC-02"])
    c.add_argument("--limit", type=int, help="descarcă doar primele N lecții SRC-01")
    c.add_argument("--cookies", help="cookies.txt YouTube (dacă YouTube cere autentificare)")
    c.add_argument("--offline", action="store_true", help="nu accesa site-ul, folosește lista inclusă")
    c.add_argument("--inventory-only", action="store_true", help="doar scanează fișierele existente")
    sub.add_parser("check-sources", help="diagnostic: sunt videourile YouTube accesibile?")
    sub.add_parser("extract").add_argument("--loudnorm", action="store_true")
    s = sub.add_parser("segment"); s.add_argument("--vad", choices=["silero", "energy"], default=None)
    l = sub.add_parser("label")
    l.add_argument("--labeler", choices=["robin", "hf", "csv"], default="robin")
    l.add_argument("--robin-url"); l.add_argument("--hf-model"); l.add_argument("--csv")
    l.add_argument("--workers", type=int)
    l.add_argument("--limit", type=int, help="etichetează doar N segmente (aleatoriu, din toate lecțiile)")
    l.add_argument("--batch", type=int, help="dimensiune batch pentru modele locale (implicit 8)")
    sub.add_parser("split")
    sub.add_parser("validate-export")
    vi = sub.add_parser("validate-import"); vi.add_argument("csv")
    sub.add_parser("validate-apply")
    sub.add_parser("datacard")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=float); t.add_argument("--max-steps", type=int)
    t.add_argument("--base-model"); t.add_argument("--resume", action="store_true")
    t.add_argument("--smoke", action="store_true", help="model minuscul aleator, fără descărcări")
    e = sub.add_parser("evaluate"); e.add_argument("--split", default="test"); e.add_argument("--model")
    sv = sub.add_parser("serve"); sv.add_argument("--host", default="127.0.0.1"); sv.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("prepare", help="extract + segment + label + split + datacard")
    p.add_argument("--labeler", choices=["robin", "hf", "csv"], default="robin")
    p.add_argument("--robin-url"); p.add_argument("--hf-model"); p.add_argument("--csv")
    sm = sub.add_parser("smoke"); sm.add_argument("--steps", type=int, default=600); sm.add_argument("--dir")

    a = ap.parse_args(argv)
    cfg = Config(base_dir=a.project_dir) if a.project_dir else Config()
    if a.cmd in ("collect", "extract", "segment", "prepare", "smoke", "label"):
        from asr.audio_io import AudioError, _need_ffmpeg
        try:
            _need_ffmpeg()
        except AudioError as e:
            sys.exit(f"❌ {e}")
    if getattr(a, "robin_url", None):
        cfg.robin_url = a.robin_url
    if getattr(a, "base_model", None):
        cfg.base_model = a.base_model

    if a.cmd == "collect":
        from asr import collect
        if a.inventory_only:
            inv = collect.build_inventory(cfg.make_dirs()); collect.save_inventory(cfg, inv)
            print(f"{len(inv)} fișiere inventariate")
        else:
            collect.collect(cfg, tuple(a.sources), a.limit, a.cookies, online=not a.offline)
    elif a.cmd == "check-sources":
        from collections import Counter
        from asr import collect
        res = collect.check_youtube([r["video_id"] for r in collect.load_bundled_list()])
        print(dict(Counter(res.values())))
        print("ok = descărcabile · 'încorporare dezactivată' = doar cu cookies · 'șters/privat' = indisponibile")
    elif a.cmd == "extract":
        from asr import collect, segment
        inv = collect.build_inventory(cfg.make_dirs()); collect.save_inventory(cfg, inv)
        segment.extract_all(cfg, inv, loudnorm=a.loudnorm)
    elif a.cmd == "segment":
        from asr import segment
        if a.vad:
            cfg.vad = a.vad
        segment.segment_all(cfg)
    elif a.cmd in ("label", "prepare"):
        if a.cmd == "prepare":
            from asr import collect, segment
            inv = collect.build_inventory(cfg.make_dirs()); collect.save_inventory(cfg, inv)
            inv = segment.extract_all(cfg, inv); segment.segment_all(cfg, inv)
        _label(cfg, a)
        if a.cmd == "prepare":
            from asr import datacard, splits
            splits.make_splits(cfg); datacard.write_data_card(cfg)
            print("\n➡ Următorii pași: validare manuală (python run_pipeline.py serve → tab Validare), "
                  "apoi: validate-apply, train, evaluate")
    elif a.cmd == "split":
        from asr import splits
        splits.make_splits(cfg)
    elif a.cmd == "validate-export":
        from asr import validate
        validate.export_sheet(cfg)
    elif a.cmd == "validate-import":
        from asr import validate
        print(f"{validate.import_sheet(cfg, a.csv)} decizii importate"); validate.apply_validation(cfg)
    elif a.cmd == "validate-apply":
        from asr import validate
        validate.apply_validation(cfg)
    elif a.cmd == "datacard":
        from asr import datacard
        datacard.write_data_card(cfg)
    elif a.cmd == "train":
        from asr import datacard
        from asr.train import train
        train(cfg, smoke=a.smoke, max_steps=a.max_steps, epochs=a.epochs, resume=a.resume)
        datacard.write_data_card(cfg)
    elif a.cmd == "evaluate":
        from asr import datacard
        from asr.evaluate import evaluate_split
        r = evaluate_split(cfg, a.split, a.model)
        print(json.dumps({k: r[k] for k in ("overall", "validated", "rtf") if k in r}, indent=2, ensure_ascii=False))
        if r.get("warning"):
            print("⚠", r["warning"])
        datacard.write_data_card(cfg)
        print(f"📄 Raport: {cfg.results_dir / ('eval_' + a.split + '.md')}")
    elif a.cmd == "serve":
        import os
        import uvicorn
        if a.project_dir:
            os.environ["ASR_PROJECT_DIR"] = a.project_dir
        uvicorn.run("server:app", host=a.host, port=a.port)
    elif a.cmd == "smoke":
        from asr.smoke import run_smoke
        d = a.dir or tempfile.mkdtemp(prefix="asr_smoke_")
        r = run_smoke(d, max_steps=a.steps)
        v = r["eval"]["validated"]
        print(f"\n✅ SMOKE OK — {d}\n   test validat: WER {v['wer']:.1%} · CER {v['cer']:.1%} (date sintetice, nu vorbire reală)")


def _label(cfg: Config, a):
    from asr import labelers
    from asr.segment import read_jsonl
    segs = read_jsonl(cfg.manifests_dir / "segments.jsonl")
    if not segs:
        sys.exit("Nu există segmente. Rulați întâi: extract, segment.")
    if a.labeler == "robin":
        lab = labelers.RobinLabeler(cfg.robin_url, cfg.robin_timeout)
    elif a.labeler == "hf":
        if not a.hf_model:
            sys.exit("--hf-model este obligatoriu (ex. openai/whisper-small sau un director cu model CTC)")
        lab = labelers.HFLabeler(a.hf_model, cfg)
    else:
        if not a.csv:
            sys.exit("--csv este obligatoriu (coloane: id,text)")
        lab = labelers.MappingLabeler(labelers.load_mapping_csv(a.csv), name="csv")
    labelers.label_segments(cfg, segs, lab, getattr(a, "workers", None),
                            save_path=cfg.manifests_dir / "segments.jsonl",
                            limit=getattr(a, "limit", None), batch=getattr(a, "batch", None))


if __name__ == "__main__":
    main()

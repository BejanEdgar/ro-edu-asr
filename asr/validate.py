"""Pasul 4 — validare manuală (10% din setul de test).

Decizile se păstrează în `manifests/validation.jsonl` (o linie per decizie, ultima câștigă) și
sunt aplicate în `test.jsonl` de `apply_validation`. Se completează fie în interfața web
(tab „Validare”, servită de server.py), fie într-un CSV deschis în Excel/LibreOffice.

De ce contează: etichetele de test vin de la Robin. Fără validare, "WER" măsoară cât de bine
imită modelul pe Robin, nu cât de corect transcrie. Doar subsetul validat dă un WER real.
"""
from __future__ import annotations

import csv
import json
import pathlib
import time

from .config import Config
from .segment import read_jsonl, write_jsonl
from .text_norm import normalize_text

STATUSES = ("confirmed", "edited", "deleted")


def validation_path(cfg: Config) -> pathlib.Path:
    return cfg.manifests_dir / "validation.jsonl"


def load_decisions(cfg: Config) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(validation_path(cfg)):
        out[r["id"]] = r          # ultima decizie per segment
    return out


def record_decision(cfg: Config, seg_id: str, text: str | None, status: str) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status invalid: {status}")
    rec = {"id": seg_id, "status": status, "text": normalize_text(text or ""), "ts": int(time.time())}
    p = validation_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def export_sheet(cfg: Config, path=None) -> pathlib.Path:
    """CSV pentru Excel: completați `corectat` și (opțional) `actiune` = sterge."""
    path = pathlib.Path(path or cfg.manifests_dir / "validation_sheet.csv")
    rows = [s for s in read_jsonl(cfg.manifests_dir / "test.jsonl") if s.get("val_selected")]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "audio_filepath", "durata_s", "pseudo_label", "corectat", "actiune"])
        for s in rows:
            w.writerow([s["id"], s["audio_filepath"], s["duration"], s.get("text_pseudo", s["text"]), "", ""])
    print(f"📄 {len(rows)} segmente → {path}  (completați coloana 'corectat'; "
          f"lăsați goală = confirmat; actiune=sterge pentru a exclude)")
    return path


def import_sheet(cfg: Config, path) -> int:
    n = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            action = (r.get("actiune") or "").strip().lower()
            fixed = (r.get("corectat") or "").strip()
            if action in ("sterge", "șterge", "delete"):
                record_decision(cfg, r["id"], "", "deleted")
            elif fixed and normalize_text(fixed) != normalize_text(r["pseudo_label"]):
                record_decision(cfg, r["id"], fixed, "edited")
            else:
                record_decision(cfg, r["id"], r["pseudo_label"], "confirmed")
            n += 1
    return n


def apply_validation(cfg: Config) -> dict:
    """Aplică deciziile în test.jsonl. `text_pseudo` rămâne neschimbat (necesar pentru a
    măsura cât de bun este profesorul Robin)."""
    p = cfg.manifests_dir / "test.jsonl"
    test = read_jsonl(p)
    dec = load_decisions(cfg)
    counts = {"confirmed": 0, "edited": 0, "deleted": 0}
    for s in test:
        d = dec.get(s["id"])
        if not d:
            continue
        counts[d["status"]] += 1
        if d["status"] == "deleted":
            s["excluded"] = True
            continue
        s["excluded"] = False
        s["validated"] = True
        s["text"] = d["text"] or s.get("text_pseudo", s["text"])
        s["edited"] = d["status"] == "edited"
    write_jsonl(p, test)
    target = sum(1 for s in test if s.get("val_selected"))
    counts.update(selected=target, done=sum(counts.values()))
    print(f"✅ Validare aplicată: {counts}")
    return counts

"""Pasul 7 — evaluarea modelului: WER, CER, tipuri de erori, viteză (RTF).

Metrici raportate separat, pentru că au semnificații diferite:
  • WER/CER pe subsetul VALIDAT MANUAL  = acuratețea reală a modelului (referință corectată de om)
  • WER/CER pe tot testul               = acord cu etichetele automate (Robin); optimist/nesigur
  • WER profesor (Robin vs. corectat)   = cât de bune erau pseudo-etichetele pe subsetul validat
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from typing import Sequence

import jiwer
import soundfile as sf

from .config import Config
from .dataset import load_split


def _clean_pairs(refs: Sequence[str], hyps: Sequence[str]):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r and r.strip()]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def wer_cer(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    refs, hyps = _clean_pairs(refs, hyps)
    if not refs:
        return {"wer": None, "cer": None, "n": 0}
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    n_words = w.hits + w.substitutions + w.deletions
    return {
        "wer": round(float(w.wer), 4), "cer": round(float(c.cer), 4), "n": len(refs),
        "ref_words": n_words, "substitutions": w.substitutions, "deletions": w.deletions,
        "insertions": w.insertions,
    }


def top_confusions(refs: Sequence[str], hyps: Sequence[str], k: int = 15) -> list[dict]:
    refs, hyps = _clean_pairs(refs, hyps)
    if not refs:
        return []
    out = jiwer.process_words(refs, hyps)
    cnt: Counter = Counter()
    for r, h, al in zip(refs, hyps, out.alignments):
        rw, hw = r.split(), h.split()
        for ch in al:
            if ch.type == "substitute":
                cnt[(" ".join(rw[ch.ref_start_idx:ch.ref_end_idx]),
                     " ".join(hw[ch.hyp_start_idx:ch.hyp_end_idx]))] += 1
    return [{"ref": a, "hyp": b, "count": n} for (a, b), n in cnt.most_common(k)]


def transcribe_segments(cfg: Config, transcriber, segments: list[dict], batch: int = 16):
    """Transcrie segmentele (batch) și măsoară RTF-ul. Returnează (texte, secunde_inferență)."""
    import numpy as np
    hyps = [""] * len(segments)
    t0 = time.perf_counter()
    for k in range(0, len(segments), batch):
        part = segments[k:k + batch]
        arrs = [sf.read(str(cfg.resolve(s["audio_filepath"])), dtype="float32")[0] for s in part]
        arrs = [a.mean(axis=1) if a.ndim > 1 else a for a in arrs]
        for i, t in enumerate(transcriber.transcribe_batch(arrs)):
            hyps[k + i] = t
    return hyps, time.perf_counter() - t0


def evaluate_split(cfg: Config, split: str = "test", model_dir=None, transcriber=None,
                   save: bool = True, max_examples: int = 8) -> dict:
    from .inference import Transcriber
    model_dir = model_dir or cfg.final_dir
    tr = transcriber or Transcriber(str(model_dir), vad="none")
    segs = [s for s in load_split(cfg, split) if not s.get("excluded") and s.get("text")]
    hyps, secs = transcribe_segments(cfg, tr, segs)
    refs = [s["text"] for s in segs]
    audio_s = sum(s["duration"] for s in segs)

    res = {
        "split": split, "model": str(model_dir), "device": tr.device, "n_segments": len(segs),
        "audio_hours": round(audio_s / 3600, 3),
        "rtf": round(secs / max(audio_s, 1e-6), 4),
        "overall": wer_cer(refs, hyps),
    }
    val = [i for i, s in enumerate(segs) if s.get("validated")]
    res["validated"] = wer_cer([refs[i] for i in val], [hyps[i] for i in val])
    res["validated"]["share_of_split"] = round(len(val) / max(len(segs), 1), 3)
    if val:
        # cât de bun era "profesorul" (Robin) față de corecturile umane
        res["teacher_on_validated"] = wer_cer([refs[i] for i in val],
                                              [segs[i].get("text_pseudo", "") for i in val])
    by_src = defaultdict(list)
    for i, s in enumerate(segs):
        by_src[s["source"]].append(i)
    res["by_source"] = {k: wer_cer([refs[i] for i in v], [hyps[i] for i in v]) for k, v in sorted(by_src.items())}
    res["top_confusions"] = top_confusions(refs, hyps)
    worst = sorted(range(len(segs)),
                   key=lambda i: -(wer_cer([refs[i]], [hyps[i]])["wer"] or 0))[:max_examples]
    res["worst_examples"] = [{"id": segs[i]["id"], "ref": refs[i], "hyp": hyps[i]} for i in worst]
    if not val:
        res["warning"] = ("Nicio transcriere din test nu a fost validată manual: WER-ul 'overall' măsoară "
                          "acordul cu pseudo-etichetele, nu acuratețea reală. Rulați pasul de validare.")
    if save:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        (cfg.results_dir / f"eval_{split}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(cfg.results_dir / f"eval_{split}_predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "validated", "referinta", "predictie", "wer_segment"])
            for s, h in zip(segs, hyps):
                w.writerow([s["id"], s["source"], bool(s.get("validated")), s["text"], h,
                            wer_cer([s["text"]], [h])["wer"]])
        write_report(cfg, res)
    return res


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def write_report(cfg: Config, res: dict):
    o, v = res["overall"], res["validated"]
    lines = [
        f"# Evaluare model — set `{res['split']}`", "",
        f"- Model: `{res['model']}` · dispozitiv: {res['device']}",
        f"- Segmente: {res['n_segments']} ({res['audio_hours']} h) · RTF: {res['rtf']}", "",
        "| Metrică | WER | CER | segmente |", "|---|---|---|---|",
        f"| Tot setul (ref. = pseudo-etichete/validate) | {_pct(o['wer'])} | {_pct(o['cer'])} | {o['n']} |",
        f"| **Subset validat manual (acuratețe reală)** | {_pct(v['wer'])} | {_pct(v['cer'])} | {v['n']} |",
    ]
    if "teacher_on_validated" in res:
        t = res["teacher_on_validated"]
        lines.append(f"| Profesor (Robin) pe subsetul validat | {_pct(t['wer'])} | {_pct(t['cer'])} | {t['n']} |")
    lines += ["", f"Erori (tot setul): substituții {o.get('substitutions')}, ștergeri {o.get('deletions')}, "
                  f"inserții {o.get('insertions')}", ""]
    if res.get("warning"):
        lines += [f"> ⚠ {res['warning']}", ""]
    lines += ["## Confuzii frecvente (referință → predicție)", "", "| ref | hyp | nr |", "|---|---|---|"]
    lines += [f"| {c['ref']} | {c['hyp']} | {c['count']} |" for c in res["top_confusions"]]
    lines += ["", "## Cele mai slabe exemple", ""]
    for e in res["worst_examples"]:
        lines += [f"- `{e['id']}`", f"  - REF : {e['ref']}", f"  - PRED: {e['hyp']}"]
    (cfg.results_dir / f"eval_{res['split']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


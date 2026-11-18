"""Pasul 2 — Data Card (JSON + Markdown), regenerat din datele existente.

Toate cifrele provin din fișierele efectiv colectate/prelucrate; ce nu există încă apare ca
„încă nu a fost calculat", nu ca valoare inventată. Se poate rula după fiecare etapă.
"""
from __future__ import annotations

import datetime
import json
from collections import Counter

from .collect import load_inventory
from .config import SOURCES, Config
from .segment import read_jsonl


def _hours(rows, key="duration"):
    return round(sum(r.get(key, 0) for r in rows) / 3600, 3)


def build_data_card(cfg: Config) -> dict:
    inv = load_inventory(cfg)
    segs = read_jsonl(cfg.manifests_dir / "segments.jsonl")
    splits = {n: read_jsonl(cfg.manifests_dir / f"{n}.jsonl") for n in ("train", "dev", "test")}
    dec = read_jsonl(cfg.manifests_dir / "validation.jsonl")
    last = {d["id"]: d for d in dec}
    summ_p = cfg.final_dir / "training_summary.json"
    eval_p = cfg.results_dir / "eval_test.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else None
    ev = json.loads(eval_p.read_text(encoding="utf-8")) if eval_p.exists() else None

    sources = []
    for sid, meta in SOURCES.items():
        files = [i for i in inv if i["source"] == sid]
        ss = [s for s in segs if s["source"] == sid]
        if not files and sid == "SRC-03":
            continue
        sources.append({
            "id": sid, "name": meta["name"], "url": meta["url"], "files": len(files),
            "recordings_hours": round(sum(f["duration_s"] for f in files) / 3600, 3),
            "speech_segments": len(ss), "speech_hours": _hours(ss),
            "speakers": len({f["speaker"] for f in files}),
        })

    labeled = [s for s in segs if s.get("text_raw") is not None]
    card = {
        "name": "RO-EduASR",
        "version": "1.1",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "description": ("Corpus audio/video educațional în limba română (lecții de matematică, clasa a 4-a) "
                        "colectat din tezaurul național, pentru antrenarea și evaluarea modelelor de "
                        "recunoaștere a vorbirii. Transcrierile sunt automate (pseudo-etichete RobinASR), "
                        "cu un subset de test corectat manual."),
        "task": "automatic-speech-recognition", "languages": ["ro"], "domain": "educație / matematică",
        "sources": sources,
        "audio_specs": {"sampling_rate_hz": cfg.sampling_rate, "channels": 1, "format": "WAV PCM 16-bit",
                        "segment_length_s": [cfg.seg_min_sec, cfg.seg_max_sec]},
        "collection": {
            "SRC-01": "linkuri YouTube listate pe educatieonline.md (audio descărcat cu yt-dlp)",
            "SRC-02": "folder Google Drive public din enunț (gdown)",
            "SRC-03": "fișiere adăugate manual (opțional)",
        },
        "preprocessing": {
            "extract": "ffmpeg -> WAV mono 16 kHz PCM16",
            "vad": cfg.vad + (" (Silero VAD)" if cfg.vad == "silero" else ""),
            "segmentation": f"{cfg.seg_min_sec}–{cfg.seg_max_sec} s; pauze <{cfg.seg_merge_gap_sec}s unite",
            "dedup": "md5 pe PCM16 al segmentului",
            "text_normalization": "lowercase, ş/ţ->ș/ț, cifre și simboluri -> litere (num2words ro), "
                                  "doar a-z ă â î ș ț spațiu",
        },
        "transcription": {
            "method": "pseudo-etichete automate; test: subset corectat manual",
            "labelers": dict(Counter(s.get("label_source", "?") for s in labeled)),
            "segments_labeled": len(labeled),
            "segments_usable": sum(1 for s in segs if s.get("usable")),
            "segments_flagged": dict(Counter(s.get("flag") for s in segs if s.get("flag"))),
        },
        "splits": {
            "policy": "pe vorbitor (profesor) / înregistrare — fără scurgeri de date între seturi",
            **{n: {"segments": len(r), "hours": _hours(r), "speakers": len({s["speaker"] for s in r}),
                   "sources": dict(Counter(s["source"] for s in r))} for n, r in splits.items()},
        },
        "manual_validation": {
            "selected": sum(1 for s in splits["test"] if s.get("val_selected")),
            "decisions": dict(Counter(d["status"] for d in last.values())),
            "validated_in_test": sum(1 for s in splits["test"] if s.get("validated")),
        },
        "statistics": {
            "recordings": len(inv), "recordings_hours": round(sum(i["duration_s"] for i in inv) / 3600, 3),
            "size_gb": round(sum(i["size_mb"] for i in inv) / 1024, 3),
            "speech_segments": len(segs), "speech_hours": _hours(segs),
        },
        "rights_and_ethics": [
            "Conținutul aparține autorilor/publicatorilor originali (lecții publicate pe educatieonline.md "
            "și YouTube). Corpusul se folosește exclusiv în scop didactic/de cercetare; nu se redistribuie.",
            "Datele conțin voci ale unor profesori identificabili; nu se publică nume în setul derivat.",
            "Licența sursei SRC-02 (Google Drive) nu este documentată în enunț — de confirmat înainte de orice publicare.",
        ],
        "limitations": [
            "Domeniu îngust (matematică clasa a 4-a): vocabular specific, fără vorbire spontană.",
            "Etichetele de antrenare sunt automate (Robin) — modelul poate învăța și erorile acestuia.",
            "Număr mic de vorbitori (profesori) -> generalizare limitată la alte voci/accente.",
            "Zgomot/muzică de fundal în unele lecții; VAD poate tăia sau păstra greșit unele pasaje.",
            "Cifrele sunt transcrise literal (ex. 25 -> douăzeci și cinci); modelul nu produce cifre.",
        ],
    }
    if summ:
        card["model"] = {k: summ[k] for k in ("base_model", "train_hours", "dev_wer", "dev_cer",
                                              "train_minutes", "trained_at") if k in summ}
    if ev:
        card["results_test"] = {"overall": ev["overall"], "validated": ev["validated"],
                                "teacher_on_validated": ev.get("teacher_on_validated"), "rtf": ev["rtf"]}
    return card


def _fmt(x):
    return "—" if x is None else (f"{x:.1%}" if isinstance(x, float) and x <= 1.5 else str(x))


def to_markdown(c: dict) -> str:
    L = [f"# Data Card — {c['name']} v{c['version']}", "",
         f"_Generat automat: {c['created_at']}_", "", c["description"], "",
         f"**Limbă:** {', '.join(c['languages'])} · **Sarcină:** {c['task']} · **Domeniu:** {c['domain']}", "",
         "## Surse", "", "| ID | Sursă | Fișiere | Ore înreg. | Segmente | Ore vorbire | Vorbitori |",
         "|---|---|---|---|---|---|---|"]
    for s in c["sources"]:
        L.append(f"| {s['id']} | [{s['name']}]({s['url']}) | {s['files']} | {s['recordings_hours']} | "
                 f"{s['speech_segments']} | {s['speech_hours']} | {s['speakers']} |" if s["url"] else
                 f"| {s['id']} | {s['name']} | {s['files']} | {s['recordings_hours']} | "
                 f"{s['speech_segments']} | {s['speech_hours']} | {s['speakers']} |")
    st = c["statistics"]
    L += ["", "## Statistici", "",
          f"- Înregistrări: **{st['recordings']}** ({st['recordings_hours']} h, {st['size_gb']} GB)",
          f"- Segmente de vorbire: **{st['speech_segments']}** ({st['speech_hours']} h)", "",
          "## Colectare", ""] + [f"- **{k}:** {v}" for k, v in c["collection"].items()]
    a, p = c["audio_specs"], c["preprocessing"]
    L += ["", "## Preprocesare", "",
          f"- Audio: {a['sampling_rate_hz']} Hz, mono, {a['format']}"] + [f"- {k}: {v}" for k, v in p.items()]
    t = c["transcription"]
    L += ["", "## Transcrieri", "", f"- Metodă: {t['method']}", f"- Etichetatori: {t['labelers']}",
          f"- Segmente etichetate / utilizabile: {t['segments_labeled']} / {t['segments_usable']}",
          f"- Eliminate automat: {t['segments_flagged'] or '—'}", "", "## Împărțire", "",
          f"Politică: {c['splits']['policy']}", "", "| Set | Segmente | Ore | Vorbitori |", "|---|---|---|---|"]
    for n in ("train", "dev", "test"):
        s = c["splits"][n]
        L.append(f"| {n} | {s['segments']} | {s['hours']} | {s['speakers']} |")
    v = c["manual_validation"]
    L += ["", "## Validare manuală", "",
          f"- Selectate: {v['selected']} · decizii: {v['decisions'] or 'încă niciuna'} · "
          f"validate în test: {v['validated_in_test']}"]
    if "model" in c:
        L += ["", "## Model antrenat", ""] + [f"- {k}: {val}" for k, val in c["model"].items()]
    if "results_test" in c:
        r = c["results_test"]
        L += ["", "## Rezultate pe test", "",
              f"- WER/CER pe subsetul validat manual: {_fmt(r['validated']['wer'])} / {_fmt(r['validated']['cer'])} "
              f"({r['validated']['n']} segmente)",
              f"- WER/CER pe tot testul: {_fmt(r['overall']['wer'])} / {_fmt(r['overall']['cer'])}",
              f"- RTF: {r['rtf']}"]
    L += ["", "## Drepturi și etică", ""] + [f"- {x}" for x in c["rights_and_ethics"]]
    L += ["", "## Limitări", ""] + [f"- {x}" for x in c["limitations"]]
    return "\n".join(L) + "\n"


def write_data_card(cfg: Config) -> dict:
    card = build_data_card(cfg)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_card_json.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg.data_card_md.write_text(to_markdown(card), encoding="utf-8")
    print(f"📋 Data Card → {cfg.data_card_json.name}, {cfg.data_card_md.name}")
    return card

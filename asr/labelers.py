"""Pasul 3 (preprocesare) — transcrieri (pseudo-etichete) pentru segmente.

Interfață: `labeler.label(seg: dict, wav_path: Path) -> str`

  RobinLabeler     RobinASR (https://github.com/racai-ai/RobinASR). Serverul real primește
                   `multipart/form-data` cu câmpul `file` (WAV) pe `POST /transcribe` și
                   răspunde `{"status": "OK", "transcription": "..."}`. (Versiunea veche
                   a notebook-ului trimitea JSON cu base64 — formatul nu era cel al serverului.)
  HFLabeler        orice model CTC antrenat local sau Whisper de pe HuggingFace.
  MappingLabeler   transcrieri furnizate de utilizator (id segment -> text) sau oracol de test.
"""
from __future__ import annotations

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .config import Config
from .segment import write_jsonl
from .text_norm import normalize_text


class LabelerError(RuntimeError):
    pass


class RobinEndpointError(LabelerError):
    """URL-ul nu este un API Robin (de ex. este pagina web RELATE, nu serverul)."""


ROBIN_HINT = (
    "Răspunsul nu este JSON Robin. URL-ul https://relate.racai.ro/index.php?path=robin/asr este "
    "interfața web, nu un API documentat. Rulați serverul Robin local:\n"
    "    docker run -p 8888:8888 racai/robinasr:cpu\n"
    "și setați ROBIN_API_URL=http://localhost:8888/transcribe  (sau --robin-url ...)."
)


class RobinLabeler:
    name = "robin"

    def __init__(self, url: str, timeout: int = 60, retries: int = 3):
        self.url, self.timeout, self.retries = url, timeout, retries

    def label(self, seg: dict, wav_path: pathlib.Path) -> str:
        import requests
        data = pathlib.Path(wav_path).read_bytes()
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, files={"file": (pathlib.Path(wav_path).name, data, "audio/wav")},
                                  timeout=self.timeout)
            except requests.RequestException as e:
                last = e; time.sleep(1.5 * (attempt + 1)); continue
            if r.status_code >= 500:
                last = LabelerError(f"HTTP {r.status_code}"); time.sleep(1.5 * (attempt + 1)); continue
            try:
                j = r.json()
            except ValueError:
                raise RobinEndpointError(f"HTTP {r.status_code}, {r.headers.get('content-type', '?')}\n{ROBIN_HINT}")
            if str(j.get("status", "OK")).lower() != "ok":
                raise LabelerError(j.get("message", "eroare Robin"))
            return (j.get("transcription") or j.get("text") or "").strip()
        raise LabelerError(f"Robin indisponibil după {self.retries} încercări: {last}")


class HFLabeler:
    """Model HuggingFace: director/ID de model CTC (Wav2Vec2) sau Whisper. Folosește GPU dacă există."""
    name = "hf"

    def __init__(self, model: str, cfg: Config, device: Optional[str] = None, batch_size: int = 8):
        import torch
        self.model_id, self.cfg, self.batch_size = model, cfg, batch_size
        use_cuda = (device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"
        if "whisper" in model.lower():
            from transformers import pipeline
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()        # avertismente inofensive din generate()
            self._pipe = pipeline("automatic-speech-recognition", model=model,
                                  device=0 if use_cuda else -1,
                                  torch_dtype=torch.float16 if use_cuda else torch.float32)
            self._tr = None
        else:
            from .inference import Transcriber
            self._tr = Transcriber(model, device="cuda" if use_cuda else "cpu", vad="none", batch_size=batch_size)
            self._pipe = None
        self.name = f"hf:{model}"
        print(f"  HFLabeler: {model} pe {'GPU' if use_cuda else 'CPU (lent pentru Whisper!)'}, batch {batch_size}")

    def transcribe_arrays(self, audios: list) -> list[str]:
        """Transcrie semnale 16 kHz (≤ 30 s fiecare)."""
        if self._tr is not None:
            return self._tr.transcribe_batch(audios)
        outs = self._pipe([{"raw": a, "sampling_rate": self.cfg.sampling_rate} for a in audios],
                          batch_size=self.batch_size,
                          generate_kwargs={"language": "romanian", "task": "transcribe"})
        return [o["text"].strip() for o in outs]

    def label_many(self, segs: list[dict], paths: list) -> list[str]:
        from .audio_io import load_audio
        return self.transcribe_arrays([load_audio(p, self.cfg.sampling_rate) for p in paths])

    def label(self, seg: dict, wav_path: pathlib.Path) -> str:
        return self.label_many([seg], [wav_path])[0]


class MappingLabeler:
    """Transcrieri din dicționar {id_segment: text} sau funcție(seg) -> text."""
    name = "mapping"

    def __init__(self, mapping: dict | Callable[[dict], str], name: str = "mapping"):
        self.mapping, self.name = mapping, name

    def label(self, seg: dict, wav_path) -> str:
        if callable(self.mapping):
            return self.mapping(seg)
        return self.mapping.get(seg["id"], "")


def load_mapping_csv(path) -> dict:
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["text"] for r in csv.DictReader(f)}


# ───────────────────────── orchestrare ─────────────────────────
def label_segments(cfg: Config, segments: list[dict], labeler, workers: int | None = None,
                   save_path: pathlib.Path | None = None, save_every: int = 10,
                   verbose: bool = True, limit: int | None = None, batch: int | None = None) -> dict:
    """Etichetează segmentele (reluabil: sare peste cele care au deja `text_raw`).
    `limit`: etichetează doar N segmente (alese aleatoriu, distribuit pe toate lecțiile) — pentru
    teste rapide sau când modelul e lent. Segmentele neetichetate sunt ignorate la split/antrenare."""
    import random
    import time
    todo = [s for s in segments if s.get("text_raw") is None]
    if limit and limit < len(todo):
        todo = random.Random(cfg.seed).sample(todo, limit)
    total = len(todo)
    ok = fail = 0
    first_err = None
    t0 = time.time()
    every = 10 if getattr(labeler, "slow", False) or hasattr(labeler, "label_many") else 100

    def progress(done):
        if verbose and (done % every == 0 or done == total):
            el = time.time() - t0
            eta = el / done * (total - done) if done else 0
            print(f"  {done}/{total} · {el/60:.1f} min scurse · rămas ≈ {eta/60:.0f} min "
                  f"({el/done:.1f} s/segment)", flush=True)

    if verbose:
        print(f"🔤 De etichetat acum: {total} (deja etichetate: {len(segments) - len([s for s in segments if s.get('text_raw') is None])}"
              f"; neetichetate în total: {len([s for s in segments if s.get('text_raw') is None])}) · labeler={labeler.name}")

    def store(seg, text):
        seg["text_raw"] = text
        seg["label_source"] = labeler.name

    done = last_saved = 0
    if hasattr(labeler, "label_many"):             # model local: batch-uri, un singur fir (modelul folosește deja toate nucleele)
        bs = batch or getattr(labeler, "batch_size", 8)
        for k in range(0, total, bs):
            part = todo[k:k + bs]
            try:
                texts = labeler.label_many(part, [cfg.resolve(s["audio_filepath"]) for s in part])
                for s_, t_ in zip(part, texts):
                    store(s_, t_); ok += 1
            except Exception as e:
                fail += len(part); first_err = first_err or str(e)
                for s_ in part:
                    s_["label_error"] = str(e)[:200]
                if ok == 0 and fail >= 3 * bs:
                    raise LabelerError(f"Primele cereri au eșuat: {first_err}")
            done += len(part)
            if save_path and done - last_saved >= save_every:
                write_jsonl(save_path, segments); last_saved = done
            progress(done)
    else:                                          # API la distanță (Robin): fire paralele
        workers = workers or cfg.label_workers
        consecutive_fail = 0
        def work(seg):
            return labeler.label(seg, cfg.resolve(seg["audio_filepath"]))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(work, s): s for s in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                seg = futs[fut]
                try:
                    store(seg, fut.result()); ok += 1; consecutive_fail = 0
                except RobinEndpointError:
                    for f in futs:
                        f.cancel()
                    raise
                except Exception as e:
                    fail += 1; consecutive_fail += 1
                    first_err = first_err or str(e)
                    seg["label_error"] = str(e)[:200]
                    if ok == 0 and consecutive_fail >= 8:
                        for f in futs:
                            f.cancel()
                        raise LabelerError(f"Primele {consecutive_fail} cereri au eșuat: {first_err}")
                if save_path and n % max(save_every, 25) == 0:
                    write_jsonl(save_path, segments)
                progress(n)
    stats = finalize_labels(cfg, segments)
    stats.update(labeled_now=ok, failed_now=fail)
    if save_path:
        write_jsonl(save_path, segments)
    if verbose:
        print(f"✅ etichetate {ok}, eșuate {fail} · utilizabile {stats['usable']}/{stats['total']} "
              f"(suspecte {stats['suspect']}, goale {stats['empty']})")
    return stats


def finalize_labels(cfg: Config, segments: list[dict]) -> dict:
    """Normalizează `text_raw` -> `text_pseudo`/`text`; marchează segmentele neutilizabile."""
    empty = suspect = usable = 0
    for s in segments:
        raw = s.get("text_raw")
        if raw is None:
            s["usable"] = False
            continue
        norm = normalize_text(raw)
        s["text_pseudo"] = norm
        if not s.get("validated"):
            s["text"] = norm
        cps = len(norm) / max(s["duration"], 1e-3)
        s["chars_per_sec"] = round(cps, 2)
        if not norm:
            empty += 1; s["usable"] = False; s["flag"] = "empty"
        elif not (cfg.min_chars_per_sec <= cps <= cfg.max_chars_per_sec):
            suspect += 1; s["usable"] = False; s["flag"] = "chars_per_sec"
        else:
            usable += 1; s["usable"] = True; s.pop("flag", None)
    return {"total": len(segments), "usable": usable, "suspect": suspect, "empty": empty}

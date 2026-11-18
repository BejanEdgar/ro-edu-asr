"""Teste automate. Rulare:  pytest -q          (rapid, ~1–2 min)
                          pytest -q -m slow  (include verificarea că modelul chiar învață)
"""
import importlib
import json
import socket
import sys
import threading
import time
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr import labelers, splits, validate                      # noqa: E402
from asr.config import Config                                    # noqa: E402
from asr.segment import _energy_spans, merge_spans, read_jsonl   # noqa: E402
from asr.text_norm import normalize_text                         # noqa: E402


# ───────────────────── normalizare text ────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("Adunăm 12 + 8 = 20.", "adunăm doisprezece plus opt egal douăzeci"),
    ("Ştiţi că 25 000?", "știți că douăzeci și cinci de mii"),
    ("50% din 200", "cincizeci la sută din două sute"),
    ("3,5 cm", "trei virgulă cinci cm"),
    ("Ce-ți place? café", "ce ți place cafe"),
    ("  ȘI  ţara ", "și țara"),
    ("", ""),
])
def test_normalize(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_only_vocab_chars():
    from asr.text_norm import VOCAB_CHARS
    out = normalize_text("Test: 99 ★ ñandú — «citat» 3€")
    assert set(out) <= set(VOCAB_CHARS)


# ───────────────────── VAD / segmentare ────────────────────────
def test_energy_vad_and_merge():
    sr = 16000
    a = np.zeros(sr * 40, np.float32)
    for s, e in [(1, 4), (10, 12), (14, 34)]:
        a[int(s * sr):int(e * sr)] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int((e - s) * sr)) / sr)
    a += np.random.default_rng(0).normal(0, 0.001, a.shape).astype(np.float32)
    spans = _energy_spans(a, sr, 20.0)
    assert len(spans) >= 3
    assert all((e - s) / sr <= 20.0 + 1e-6 for s, e in spans)          # nimic peste maxim
    assert len(merge_spans(spans, sr, 20.0, 0.6)) <= len(spans)


# ───────────────────── împărțirea datelor ──────────────────────
def _fake_segments(n_speakers=8, per=12):
    segs = []
    for i in range(n_speakers):
        for j in range(per):
            segs.append({"id": f"s{i}_{j}", "recording_id": f"r{i}", "speaker": f"spk{i}", "source": "SRC-01",
                         "start": j * 5.0, "end": j * 5.0 + 4.0, "duration": 4.0, "usable": True})
    return segs


def test_group_split_has_no_speaker_leakage():
    sp = splits.group_split(_fake_segments(), (0.8, 0.1, 0.1), seed=1)
    spk = {k: {s["speaker"] for s in v} for k, v in sp.items()}
    assert spk["train"].isdisjoint(spk["test"]) and spk["train"].isdisjoint(spk["dev"]) and spk["dev"].isdisjoint(spk["test"])
    assert all(len(v) > 0 for v in sp.values())
    assert sum(len(v) for v in sp.values()) == 96


def test_split_single_recording_chronological():
    segs = [dict(s, speaker="x", recording_id="x") for s in _fake_segments(1, 30)]
    sp = splits.group_split(segs, seed=1)
    assert max(s["start"] for s in sp["train"]) < min(s["start"] for s in sp["test"])


def test_select_validation_fraction():
    test = _fake_segments(3, 40)
    picked = splits.select_validation(test, 0.10, 5, seed=1)
    assert len(picked) == 12 and sum(s["val_selected"] for s in test) == 12


# ───────────────────── client Robin (server simulat) ───────────
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_robin():
    """Server care respectă protocolul real RobinASR: POST /transcribe, multipart `file` -> JSON."""
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse
    app = FastAPI()
    calls = {"n": 0}

    @app.post("/transcribe")
    def tr(file: UploadFile = File(...)):
        calls["n"] += 1
        data = file.file.read()
        return {"status": "OK", "transcription": f"Test {len(data) > 44} 12"}

    @app.get("/webui", response_class=HTMLResponse)
    @app.post("/webui", response_class=HTMLResponse)
    def webui():
        return "<html>RELATE web interface</html>"

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True); th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield {"url": f"http://127.0.0.1:{port}", "calls": calls}
    server.should_exit = True


def test_robin_client_protocol(mock_robin, tmp_path):
    from asr.audio_io import write_wav
    wav = tmp_path / "a.wav"; write_wav(wav, np.zeros(16000, np.float32) + 0.01, 16000)
    lab = labelers.RobinLabeler(mock_robin["url"] + "/transcribe", timeout=10)
    assert lab.label({}, wav) == "Test True 12"
    assert mock_robin["calls"]["n"] >= 1


def test_robin_html_endpoint_gives_clear_error(mock_robin, tmp_path):
    from asr.audio_io import write_wav
    wav = tmp_path / "a.wav"; write_wav(wav, np.zeros(16000, np.float32), 16000)
    lab = labelers.RobinLabeler(mock_robin["url"] + "/webui", timeout=10)
    with pytest.raises(labelers.RobinEndpointError) as e:
        lab.label({}, wav)
    assert "docker" in str(e.value).lower()


def test_labeling_pipeline_with_robin(mock_robin, tmp_path):
    from asr.audio_io import write_wav
    cfg = Config(base_dir=tmp_path).make_dirs()
    segs = []
    for i in range(6):
        p = cfg.segments_dir / "r" / f"s{i}.wav"; write_wav(p, np.zeros(32000, np.float32) + 0.05, 16000)
        segs.append({"id": f"s{i}", "audio_filepath": cfg.rel(p), "duration": 2.0})
    st = labelers.label_segments(cfg, segs, labelers.RobinLabeler(mock_robin["url"] + "/transcribe"), verbose=False)
    assert st["usable"] == 6
    assert segs[0]["text_pseudo"] == "test true doisprezece"       # cifrele sunt normalizate în litere


# ───────────────────── pipeline complet (date sintetice) ───────
@pytest.fixture(scope="module")
def smoke_project(tmp_path_factory):
    from asr.smoke import run_smoke
    d = tmp_path_factory.mktemp("smoke")
    res = run_smoke(d, max_steps=30)
    return d, res


def test_smoke_artifacts(smoke_project):
    d, res = smoke_project
    for f in ("manifests/inventory.json", "manifests/segments.jsonl", "manifests/train.jsonl",
              "manifests/test.jsonl", "data_card.json", "DATA_CARD.md", "model_final/model.safetensors",
              "model_final/vocab.json", "model_final/training_summary.json", "results/eval_test.json",
              "results/eval_test.md"):
        assert (d / f).exists(), f
    assert res["segments"] > 100
    tr, te = {s["speaker"] for s in read_jsonl(d / "manifests/train.jsonl")}, \
             {s["speaker"] for s in read_jsonl(d / "manifests/test.jsonl")}
    assert tr.isdisjoint(te)


def test_data_card_has_real_numbers(smoke_project):
    d, _ = smoke_project
    card = json.loads((d / "data_card.json").read_text(encoding="utf-8"))
    assert card["statistics"]["speech_segments"] > 100
    assert card["manual_validation"]["validated_in_test"] > 0
    assert "Data Card" in (d / "DATA_CARD.md").read_text(encoding="utf-8")


def test_validation_flow(smoke_project):
    d, _ = smoke_project
    cfg = Config(base_dir=d)
    test = read_jsonl(d / "manifests/test.jsonl")
    sel = [s for s in test if s.get("val_selected")]
    assert sel and all(s["validated"] for s in sel)
    validate.record_decision(cfg, sel[0]["id"], "Doi plus 3", "edited")
    validate.record_decision(cfg, sel[1]["id"], "", "deleted")
    c = validate.apply_validation(cfg)
    assert c["edited"] == 1 and c["deleted"] == 1
    after = {s["id"]: s for s in read_jsonl(d / "manifests/test.jsonl")}
    assert after[sel[0]["id"]]["text"] == "doi plus trei"
    assert after[sel[0]["id"]]["text_pseudo"] == sel[0]["text_pseudo"]      # pseudo-eticheta se păstrează
    assert after[sel[1]["id"]]["excluded"] is True


def test_sheet_roundtrip(smoke_project):
    d, _ = smoke_project
    cfg = Config(base_dir=d)
    p = validate.export_sheet(cfg)
    assert p.exists() and len(p.read_text(encoding="utf-8-sig").splitlines()) > 2


# ───────────────────── server ──────────────────────────────────
@pytest.fixture(scope="module")
def client(smoke_project, mock_robin):
    d, _ = smoke_project
    import os
    os.environ.update(ASR_PROJECT_DIR=str(d), MODEL_DIR=str(d / "model_final"), ASR_VAD="energy",
                      ROBIN_API_URL=mock_robin["url"] + "/transcribe")
    import server
    importlib.reload(server)
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c, d


def _a_wav(d):
    return sorted((d / "audio_raw").glob("*.wav"))[1]


def test_server_health_and_index(client):
    c, _ = client
    h = c.get("/health").json()
    assert h["model_ro_loaded"] and h["ffmpeg"]
    assert c.get("/").status_code == 200 and "ASR" in c.get("/").text


def test_server_transcribe_local_formats(client):
    c, d = client
    import subprocess
    w = _a_wav(d)
    webm = d / "rec.webm"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(w), "-c:a", "libopus", str(webm)], check=True)
    for path, mime in ((w, "audio/wav"), (webm, "audio/webm")):        # webm = ce produce microfonul din browser
        r = c.post("/transcribe", files={"file": (path.name, path.read_bytes(), mime)},
                   data={"language": "ro", "engine": "local"})
        assert r.status_code == 200, r.text
        j = r.json()
        # modelul din fixture e antrenat doar 30 de pași (text gol e normal); verificăm fluxul, nu calitatea
        assert j["engine"] == "local" and j["duration_audio"] > 30 and isinstance(j["segments"], list) and j["rtf"] < 1


def test_server_robin_engine_via_vad_chunks(client):
    c, d = client
    w = _a_wav(d)
    r = c.post("/transcribe", files={"file": ("a.wav", w.read_bytes())}, data={"engine": "robin"})
    assert r.status_code == 200 and r.json()["engine"] == "robin"
    assert r.json()["transcription"].startswith("test true")


def test_server_errors_are_clear(client):
    c, _ = client
    r = c.post("/transcribe", files={"file": ("x.txt", b"not audio")}, data={})
    assert r.status_code == 400
    assert c.post("/transcribe", files={"file": ("x.wav", b"")}, data={"language": "xx"}).status_code in (400,)


def test_server_base64_batch_and_card(client):
    import base64
    c, d = client
    w = _a_wav(d)
    r = c.post("/transcribe/base64", json={"audio": base64.b64encode(w.read_bytes()).decode(), "format": "wav", "engine": "local"})
    assert r.status_code == 200
    r = c.post("/transcribe/batch", files=[("files", ("a.wav", w.read_bytes())), ("files", ("b.txt", b"zz"))],
               data={"engine": "local"})
    assert [x["status"] for x in r.json()["results"]] == ["ok", "error"]
    assert "data_card" in c.get("/model-card").json()


def test_server_validation_api_and_path_traversal(client):
    c, _ = client
    segs = c.get("/api/manifest/test?selected=true").json()
    assert segs
    sid = segs[0]["id"]
    assert c.get(f"/api/segment_audio/{sid}").status_code == 200
    assert c.get("/api/segment_audio/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert c.post("/api/validation", json={"id": sid, "status": "bogus", "text": ""}).status_code == 400
    assert c.post("/api/validation", json={"id": "nu_exista", "status": "confirmed", "text": ""}).status_code == 404


# ───────────────────── modelul chiar învață (lent) ─────────────
@pytest.mark.slow
def test_training_actually_learns(tmp_path):
    """Pe codul audio sintetic, după ~450 pași CER-ul trebuie să fie mult sub 100% (platoul CTC)."""
    from asr.smoke import run_smoke
    res = run_smoke(tmp_path, max_steps=450)
    assert res["eval"]["validated"]["cer"] < 0.4, res["eval"]["validated"]

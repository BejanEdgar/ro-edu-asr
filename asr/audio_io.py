"""Intrare/ieșire audio. Totul trece prin ffmpeg: același cod citește WAV, MP3, MP4,
WebM (înregistrările din browser sunt WebM/Opus, nu WAV!), MKV etc."""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import soundfile as sf

from .config import AUDIO_EXT, VIDEO_EXT


class AudioError(RuntimeError):
    pass


def _autodetect_ffmpeg() -> None:
    """Windows: dacă ffmpeg nu e în PATH, îl caută în locurile uzuale (C:\\ffmpeg, FFMPEG_DIR)
    și îi adaugă folderul în PATH-ul acestui proces."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    roots = [os.getenv("FFMPEG_DIR", ""), r"C:\ffmpeg", r"C:\Program Files\ffmpeg",
             os.path.join(os.path.expanduser("~"), "ffmpeg")]
    for root in filter(None, roots):
        r = pathlib.Path(root)
        if not r.exists():
            continue
        for exe in r.rglob("ffmpeg.exe"):
            if (exe.parent / "ffprobe.exe").exists():
                os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
                return


_autodetect_ffmpeg()


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _need_ffmpeg():
    if not have_ffmpeg():
        raise AudioError(
            "ffmpeg/ffprobe nu sunt găsite în PATH. Instalare: Linux: sudo apt install ffmpeg · "
            "Windows: descărcați https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip, dezarhivați în "
            "C:\\ffmpeg (sau setați FFMPEG_DIR) și deschideți un terminal NOU.")


def load_audio(path, sr: int = 16_000) -> np.ndarray:
    """Decodează orice fișier audio/video -> float32 mono la `sr` Hz."""
    _need_ffmpeg()
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise AudioError(f"ffmpeg nu a putut decoda {pathlib.Path(path).name}: "
                         f"{p.stderr.decode(errors='ignore').strip()[:300]}")
    audio = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioError(f"{pathlib.Path(path).name}: nu conține pistă audio")
    return audio


def probe_duration(path) -> float:
    """Durata în secunde (0.0 dacă nu poate fi citită)."""
    _need_ffmpeg()
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        return float(json.loads(p.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def has_video_stream(path) -> bool:
    _need_ffmpeg()
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_type", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        return bool(json.loads(p.stdout).get("streams"))
    except Exception:
        return False


def extract_audio(src, dst, sr: int = 16_000, loudnorm: bool = False, overwrite: bool = False):
    """Video/audio -> WAV mono PCM16 la `sr` Hz. Returnează calea WAV."""
    _need_ffmpeg()
    dst = pathlib.Path(dst)
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(sr)]
    if loudnorm:
        cmd += ["-af", "loudnorm"]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        dst.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg a eșuat pentru {pathlib.Path(src).name}: {p.stderr.strip()[:300]}")
    return dst


def write_wav(path, audio: np.ndarray, sr: int = 16_000):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) if audio.size else 0.0
    return -120.0 if rms <= 1e-9 else 20.0 * np.log10(rms)


def is_media(path) -> bool:
    return pathlib.Path(path).suffix.lower() in (VIDEO_EXT | AUDIO_EXT)

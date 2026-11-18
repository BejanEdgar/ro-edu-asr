"""Pasul 1 — colectarea setului de date.

Surse:
  SRC-01  educatieonline.md (clasa 4, Matematică). Pagina conține linkuri YouTube
          (nu fișiere MP4 și nu playere încorporate), deci nu este nevoie de Selenium:
          extragem id-urile cu `requests` + regex, cu lista inclusă în
          `data/educatieonline_cl4_mate.csv` ca rezervă (site-ul poate fi indisponibil).
  SRC-02  folderul Google Drive din enunț (gdown).
  SRC-03  fișiere puse manual în asr_project/videos/manual/.

Audio-ul din videoclipuri este tot ce are nevoie ASR-ul, așa că de pe YouTube se
descarcă doar pista audio (mult mai mic decât videoclipul).
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
from typing import Iterable

from .audio_io import has_video_stream, is_media, probe_duration
from .config import GDRIVE_FOLDER_ID, ROOT, SOURCE_URL_RO, SOURCES, Config

BUNDLED_LIST = ROOT / "data" / "educatieonline_cl4_mate.csv"
_YT_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")


# ───────────────────────── SRC-01 ──────────────────────────────
def load_bundled_list(path=BUNDLED_LIST) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scrape_youtube_ids(url: str = SOURCE_URL_RO, timeout: int = 20) -> list[str]:
    """Extrage id-urile YouTube din pagina sursă (HTML randat de server)."""
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (RO-EduASR lab)"})
    r.raise_for_status()
    seen, ids = set(), []
    for vid in _YT_RE.findall(r.text):
        if vid not in seen:
            seen.add(vid); ids.append(vid)
    return ids


def get_source1_list(online: bool = True) -> list[dict]:
    """Lista lecțiilor SRC-01: din site dacă se poate, completată cu lista inclusă."""
    bundled = {r["video_id"]: r for r in load_bundled_list()}
    ids: list[str] = []
    if online:
        try:
            ids = scrape_youtube_ids()
            print(f"  site: {len(ids)} linkuri YouTube găsite")
        except Exception as e:  # rețea / structură pagină schimbată
            print(f"  ⚠ site indisponibil ({type(e).__name__}) — folosesc lista inclusă")
    for vid in bundled:
        if vid not in ids:
            ids.append(vid)
    return [bundled.get(v, {"video_id": v, "title": "", "speaker": "", "lesson_page": ""}) for v in ids]


def check_youtube(ids: list[str], timeout: int = 15) -> dict[str, str]:
    """Diagnostic rapid prin oEmbed: de ce nu se pot descărca videourile?
       ok = public și încorporabil · 401 = încorporarea dezactivată (merge doar pe youtube.com)
       404 = șters/privat/indisponibil · altceva = rețea/blocare."""
    import requests
    out = {}
    for vid in ids:
        try:
            r = requests.get("https://www.youtube.com/oembed", timeout=timeout,
                             params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"})
            out[vid] = {200: "ok", 401: "încorporare dezactivată", 403: "blocat/restricționat",
                        404: "șters / privat / indisponibil"}.get(r.status_code, f"HTTP {r.status_code}")
        except Exception as e:
            out[vid] = f"rețea: {type(e).__name__}"
    return out


def _ytdlp_age_days() -> int | None:
    try:
        import datetime
        import yt_dlp.version as v
        y, m, d = (int(x) for x in v.__version__.split(".")[:3])
        return (datetime.date.today() - datetime.date(y, m, d)).days
    except Exception:
        return None


class _Quiet:
    """yt-dlp își tipărește singur erorile; le colectăm noi, grupate."""
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


def _clean_reason(msg: str) -> str:
    msg = re.sub(r"^(ERROR:\s*)?(\[youtube\]\s*)?[\w-]{11}:\s*", "", msg.strip().splitlines()[0])
    return re.split(r"\s*\(caused by|;\s*please report", msg)[0][:140]


def _try_download(vid: str, out_dir: pathlib.Path, cookies: str | None) -> tuple[bool, str]:
    """Încearcă mai multe strategii (YouTube refuză diferit în funcție de client/referer)."""
    import yt_dlp
    base = {"format": "bestaudio/best", "outtmpl": str(out_dir / "SRC01_%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True, "retries": 3,
            "nooverwrites": True, "logger": _Quiet()}
    if cookies:
        base["cookiefile"] = cookies
    ref = {"Referer": "https://educatieonline.md/"}
    attempts = [
        ("watch", f"https://www.youtube.com/watch?v={vid}", {}),
        ("watch+referer+embed-client", f"https://www.youtube.com/watch?v={vid}",
         {"http_headers": ref, "extractor_args": {"youtube": {"player_client": ["web_embedded", "web"]}}}),
        ("embed+referer", f"https://www.youtube.com/embed/{vid}", {"http_headers": ref}),
        ("clienți mobili", f"https://www.youtube.com/watch?v={vid}",
         {"extractor_args": {"youtube": {"player_client": ["ios", "android", "tv"]}}}),
    ]
    last = ""
    for name, url, extra in attempts:
        try:
            with yt_dlp.YoutubeDL({**base, **extra}) as ydl:
                ydl.download([url])
            if list(out_dir.glob(f"SRC01_{vid}.*")):
                return True, name
            last = "fișier negăsit după descărcare"
        except Exception as e:
            last = _clean_reason(str(e))
    return False, last


def download_source1(out_dir: pathlib.Path, limit: int | None = None, cookies: str | None = None,
                     online: bool = True) -> dict:
    """Descarcă audio-ul lecțiilor SRC-01 cu yt-dlp. Reia de unde a rămas (sare peste existente)."""
    from collections import Counter
    out_dir.mkdir(parents=True, exist_ok=True)
    items = get_source1_list(online)
    if limit:
        items = items[:limit]
    ok, failed, reasons = [], [], Counter()
    for n, it in enumerate(items, 1):
        vid = it["video_id"]
        if list(out_dir.glob(f"SRC01_{vid}.*")):
            ok.append(vid); continue
        good, info = _try_download(vid, out_dir, cookies)
        if good:
            ok.append(vid); print(f"  [{n}/{len(items)}] {vid} ✓ ({info})")
        else:
            failed.append(vid); reasons[info] += 1
            if len(failed) <= 3:
                print(f"  [{n}/{len(items)}] {vid} ✗ {info}")
    print(f"  SRC-01: {len(ok)} descărcate/existente, {len(failed)} eșuate")
    if failed:
        print("  Motive (yt-dlp):", dict(reasons))
        age = _ytdlp_age_days()
        if age is not None and age > 45:
            print(f"  ⚠ yt-dlp are {age} zile — YouTube se schimbă des. Rulați: pip install -U yt-dlp")
        diag = check_youtube(failed[:3])
        print("  Diagnostic YouTube (oEmbed, primele 3):", diag)
        if any(v == "ok" for v in diag.values()):
            print("  → Videourile EXISTĂ, deci problema e yt-dlp/rețea: pip install -U yt-dlp, sau --cookies cookies.txt "
                  "(export din browser cu extensia „Get cookies.txt LOCALLY”), sau VPN dacă e blocare regională.")
        elif any("încorporare" in v for v in diag.values()):
            print("  → Încorporarea e dezactivată: merg doar cu --cookies sau descărcare manuală din browser.")
        elif any("șters" in v for v in diag.values()):
            print("  → Videourile par șterse/private sau blocate în regiunea dvs. Folosiți folderul Google Drive (SRC-02) "
                  "sau fișiere proprii în videos/manual/.")
        print(f"  Alternativ: descărcați manual (ex. cu 4K Video Downloader) și puneți fișierele ca "
              f"SRC01_<id>.mp4 în {out_dir}")
        (out_dir.parent.parent / "manifests").mkdir(parents=True, exist_ok=True)
        (out_dir.parent.parent / "manifests" / "download_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    return {"ok": ok, "failed": failed, "reasons": dict(reasons)}


# ───────────────────────── SRC-02 ──────────────────────────────
def download_gdrive(out_dir: pathlib.Path, folder_id: str = GDRIVE_FOLDER_ID) -> dict:
    """Descarcă folderul Google Drive din enunț (trebuie să fie public). Compatibil cu versiuni
    vechi de gdown (care nu au `remaining_ok`)."""
    import inspect
    import gdown
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"output": str(out_dir), "quiet": False}
    params = inspect.signature(gdown.download_folder).parameters
    if "remaining_ok" in params:
        kw["remaining_ok"] = True          # folderele cu >50 fișiere
    if "use_cookies" in params:
        kw["use_cookies"] = False
    try:
        files = gdown.download_folder(id=folder_id, **kw) if "id" in params else \
            gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", **kw)
        n = len(files or [])
        print(f"  SRC-02: {n} fișiere descărcate")
        if n == 0:
            print("  ⚠ 0 fișiere. Posibil limita Google (prea multe descărcări) sau folder nepublic.")
        return {"ok": n, "error": None}
    except Exception as e:
        print(f"  ⚠ Google Drive: {e}\n  ℹ Încercați: pip install -U gdown. Sau deschideți folderul în browser, "
              f"Download, și dezarhivați în {out_dir}, apoi rulați din nou (inventarul le va găsi).")
        return {"ok": 0, "error": str(e)}


# ──────────────────────── inventar ─────────────────────────────
def sanitize(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:80] or "file"


def _source_of(path: pathlib.Path, videos_dir: pathlib.Path) -> str:
    try:
        top = path.resolve().relative_to(videos_dir.resolve()).parts[0]
    except (ValueError, IndexError):
        return "SRC-03"
    for sid, meta in SOURCES.items():
        if meta["subdir"] == top:
            return sid
    return "SRC-03"


def build_inventory(cfg: Config) -> list[dict]:
    """Scanează videos/ și produce inventarul (sursă, durată, dimensiune, vorbitor)."""
    meta = {r["video_id"]: r for r in load_bundled_list()}
    items = []
    for p in sorted(cfg.videos_dir.rglob("*")):
        if not p.is_file() or not is_media(p) or p.name.startswith("."):
            continue
        src = _source_of(p, cfg.videos_dir)
        prefix = SOURCES[src]["prefix"]
        stem = sanitize(p.stem)
        rid = stem if stem.startswith(prefix + "_") else f"{prefix}_{stem}"
        ytid = rid.split("_", 1)[1] if src == "SRC-01" else ""
        m = meta.get(ytid, {})
        dur = probe_duration(p)
        items.append({
            "recording_id": rid,
            "source": src,
            "file": cfg.rel(p),
            "duration_s": round(dur, 2),
            "size_mb": round(p.stat().st_size / 1e6, 2),
            "has_video": has_video_stream(p),
            "title": m.get("title", ""),
            "speaker": m.get("speaker", "") or rid,   # grup pentru split: vorbitor, altfel înregistrare
            "url": (f"https://www.youtube.com/watch?v={ytid}" if m else ""),
        })
    return items


def save_inventory(cfg: Config, items: Iterable[dict]) -> pathlib.Path:
    items = list(items)
    cfg.manifests_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.manifests_dir / "inventory.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_inventory(cfg: Config) -> list[dict]:
    p = cfg.manifests_dir / "inventory.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def collect(cfg: Config, sources=("SRC-01", "SRC-02"), limit: int | None = None,
            cookies: str | None = None, online: bool = True) -> list[dict]:
    cfg.make_dirs()
    if "SRC-01" in sources:
        print("⬇ SRC-01 educatieonline.md")
        download_source1(cfg.videos_dir / SOURCES["SRC-01"]["subdir"], limit, cookies, online)
    if "SRC-02" in sources:
        print("⬇ SRC-02 Google Drive")
        download_gdrive(cfg.videos_dir / SOURCES["SRC-02"]["subdir"])
    inv = build_inventory(cfg)
    save_inventory(cfg, inv)
    tot = sum(i["duration_s"] for i in inv) / 3600
    print(f"✅ Inventar: {len(inv)} fișiere, {tot:.2f} ore → manifests/inventory.json")
    if not inv:
        print("⚠ Niciun fișier. Puneți manual MP4/WAV în asr_project/videos/manual/ și rulați din nou.")
    return inv

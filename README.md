# Lab 4 — Modele de recunoaștere a vorbirii (RO-EduASR)

Pipeline complet: **colectare date → Data Card → preprocesare → train/test + validare manuală → model Wav2Vec2-CTC → antrenare → evaluare → server + interfață web**.
Strategia, analiza problemei și soluțiile alternative sunt în [`RAPORT.md`](RAPORT.md).

## Pornire rapidă

```bash
# 1. ffmpeg (obligatoriu):   Linux: sudo apt install ffmpeg   ·   Windows: winget install ffmpeg
# 2. dependențe (Python 3.10+; pe GPU instalați întâi PyTorch de pe pytorch.org)
pip install -r requirements.txt

# 3. verificare că TOTUL funcționează (date sintetice, fără rețea/GPU, ~2 min)
python run_pipeline.py smoke
pytest -q                                 # 25 de teste automate

# 4. rularea reală
python run_pipeline.py collect            # SRC-01 (educatieonline.md) + SRC-02 (Google Drive)
python run_pipeline.py prepare            # extract + VAD + etichetare (Robin) + split + Data Card
python run_pipeline.py serve              # → http://localhost:8000 → tab „Validare Manuală” (10% din test)
python run_pipeline.py validate-apply     # aplică corecturile în test.jsonl
python run_pipeline.py train              # necesită GPU (≥16 GB) pentru XLSR-53
python run_pipeline.py evaluate           # → asr_project/results/eval_test.md
```

Alternativ: `ro_asr_pipeline.ipynb` (aceiași pași, cu explicații și grafice). Pe Colab: încărcați proiectul, `%cd Lab4`, `!pip install -r requirements.txt`, Runtime → GPU.

## Robin (pseudo-etichete)

URL-ul din enunț (`relate.racai.ro/index.php?path=robin/asr`) este **interfața web**, nu un API documentat. Serverul RobinASR (repo `racai-ai/RobinASR`) primește `POST /transcribe` cu `multipart/form-data`, câmpul **`file`** (WAV), și răspunde `{"status": "OK", "transcription": "..."}`. Cel mai sigur:

```bash
docker run -p 8888:8888 --net=host --ipc=host racai/robinasr:cpu
export ROBIN_API_URL=http://localhost:8888/transcribe       # Windows: set ROBIN_API_URL=...
python run_pipeline.py label --labeler robin
```
Dacă URL-ul nu returnează JSON Robin, programul se oprește cu un mesaj explicit. Alternative la Robin: `--labeler hf --hf-model openai/whisper-small` sau `--labeler csv --csv transcrieri.csv` (coloane `id,text`).

## Structura

| Fișier | Rol |
|---|---|
| `run_pipeline.py` | CLI pentru toți pașii (`python run_pipeline.py -h`) |
| `ro_asr_pipeline.ipynb` | Notebook: strategie, analiză, pașii 1–7, grafice |
| `server.py` + `asr_interface.html` | API FastAPI (`/docs`) + interfață web: transcriere, validare, istoric |
| `asr/collect.py` | Colectare: YouTube (yt-dlp), Google Drive (gdown), inventar |
| `asr/segment.py` | ffmpeg → WAV 16 kHz, VAD Silero, segmente 1–20 s, deduplicare |
| `asr/text_norm.py` | Normalizare text (ș/ț, cifre → litere, simboluri) |
| `asr/labelers.py` | Robin / model HF / CSV propriu |
| `asr/splits.py`, `asr/validate.py` | Split pe vorbitor, validare manuală |
| `asr/dataset.py`, `asr/train.py` | Vocabular, Dataset, collator CTC, antrenare |
| `asr/evaluate.py`, `asr/inference.py` | WER/CER/confuzii/RTF; inferență pe audio lung |
| `asr/datacard.py` | Data Card (`asr_project/DATA_CARD.md` + `data_card.json`) |
| `data/educatieonline_cl4_mate.csv` | Cele 47 de lecții (id YouTube, titlu, profesor) |
| `tests/` | 25 de teste automate (pytest) |

Date și rezultate: `asr_project/` (`videos/`, `audio_raw/`, `segments/`, `manifests/*.jsonl`, `model_final/`, `results/`).

## Server

```bash
python server.py     # sau: python run_pipeline.py serve
curl -F file=@lectie.mp4 -F language=ro -F engine=auto http://localhost:8000/transcribe
```
Variabile: `MODEL_DIR`, `ROBIN_API_URL`, `RU_MODEL_ID`, `MAX_AUDIO_SEC`, `ASR_VAD` (`silero`|`energy`), `FALLBACK_HF_MODEL`. Fără model antrenat, `engine=robin` rămâne funcțional; `engine=auto` alege singur.

## Probleme frecvente

| Simptom | Cauză / soluție |
|---|---|
| YouTube: „Sign in to confirm you’re not a bot” | `collect --cookies cookies.txt` (export din browser) sau puneți manual fișierele `SRC01_<id>.mp4` în `asr_project/videos/educatieonline/` |
| Google Drive nu se descarcă | Descărcați folderul din browser și dezarhivați-l în `asr_project/videos/gdrive/`, apoi `collect --inventory-only` |
| `CUDA out of memory` | În `asr/config.py`: `batch_size=2`, `grad_accum=16`; sau `BASE_MODEL=facebook/wav2vec2-xls-r-300m` (aceeași dimensiune) |
| Răspuns Robin nu e JSON | Vezi secțiunea Robin: folosiți serverul local Docker |
| WER pe test „prea bun” | Fără validare manuală, WER măsoară acordul cu Robin; raportați WER-ul de pe subsetul validat |
| Rusă | Model HF `bond005/wav2vec2-large-ru-golos` (necesită rețea). DeepSpeech-ul din enunț cere Python ≤3.9 + TensorFlow vechi |

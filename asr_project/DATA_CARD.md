# Data Card — RO-EduASR v1.1

_Generat automat: 2026-11-11T12:36:00_

Corpus audio/video educațional în limba română (lecții de matematică, clasa a 4-a) colectat din tezaurul național, pentru antrenarea și evaluarea modelelor de recunoaștere a vorbirii. Transcrierile sunt automate (pseudo-etichete RobinASR), cu un subset de test corectat manual.

**Limbă:** ro · **Sarcină:** automatic-speech-recognition · **Domeniu:** educație / matematică

## Surse

| ID | Sursă | Fișiere | Ore înreg. | Segmente | Ore vorbire | Vorbitori |
|---|---|---|---|---|---|---|
| SRC-01 | [EducațieOnline.md — clasa a 4-a, Matematică (lecții video, YouTube)](https://educatieonline.md/Video?class=4&discipline=6) | 45 | 10.299 | 3758 | 8.582 | 11 |
| SRC-02 | [Google Drive — materiale video furnizate în enunțul lucrării](https://drive.google.com/drive/folders/1-7Zvo4p4ilABJVmT5izZrkpyT1P6Slvm) | 30 | 19.453 | 4942 | 18.65 | 30 |

## Statistici

- Înregistrări: **75** (29.752 h, 20.715 GB)
- Segmente de vorbire: **8700** (27.232 h)

## Colectare

- **SRC-01:** linkuri YouTube listate pe educatieonline.md (audio descărcat cu yt-dlp)
- **SRC-02:** folder Google Drive public din enunț (gdown)
- **SRC-03:** fișiere adăugate manual (opțional)

## Preprocesare

- Audio: 16000 Hz, mono, WAV PCM 16-bit
- extract: ffmpeg -> WAV mono 16 kHz PCM16
- vad: silero (Silero VAD)
- segmentation: 1.0–20.0 s; pauze <0.6s unite
- dedup: md5 pe PCM16 al segmentului
- text_normalization: lowercase, ş/ţ->ș/ț, cifre și simboluri -> litere (num2words ro), doar a-z ă â î ș ț spațiu

## Transcrieri

- Metodă: pseudo-etichete automate; test: subset corectat manual
- Etichetatori: {'hf:openai/whisper-small': 300}
- Segmente etichetate / utilizabile: 300 / 299
- Eliminate automat: {'chars_per_sec': 1}

## Împărțire

Politică: pe vorbitor (profesor) / înregistrare — fără scurgeri de date între seturi

| Set | Segmente | Ore | Vorbitori |
|---|---|---|---|
| train | 233 | 0.706 | 34 |
| dev | 28 | 0.103 | 5 |
| test | 38 | 0.097 | 2 |

## Validare manuală

- Selectate: 30 · decizii: încă niciuna · validate în test: 0

## Drepturi și etică

- Conținutul aparține autorilor/publicatorilor originali (lecții publicate pe educatieonline.md și YouTube). Corpusul se folosește exclusiv în scop didactic/de cercetare; nu se redistribuie.
- Datele conțin voci ale unor profesori identificabili; nu se publică nume în setul derivat.
- Licența sursei SRC-02 (Google Drive) nu este documentată în enunț — de confirmat înainte de orice publicare.

## Limitări

- Domeniu îngust (matematică clasa a 4-a): vocabular specific, fără vorbire spontană.
- Etichetele de antrenare sunt automate (Robin) — modelul poate învăța și erorile acestuia.
- Număr mic de vorbitori (profesori) -> generalizare limitată la alte voci/accente.
- Zgomot/muzică de fundal în unele lecții; VAD poate tăia sau păstra greșit unele pasaje.
- Cifrele sunt transcrise literal (ex. 25 -> douăzeci și cinci); modelul nu produce cifre.

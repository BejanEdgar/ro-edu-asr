"""RO-EduASR — pipeline de recunoaștere a vorbirii (română / rusă).

Module (în ordinea pașilor din lucrare):
    collect      1. colectarea setului de date (educatieonline.md, Google Drive, fișiere locale)
    datacard     2. Data Card (JSON + Markdown), regenerat automat din date
    audio_io     3. preprocesare: extragere audio din video -> WAV mono 16 kHz
    segment      3. preprocesare: segmentare VAD
    text_norm    3. preprocesare: normalizare text (diacritice, cifre -> litere)
    labelers     3. transcrieri (RobinASR / model HF / fișier propriu)
    splits       4. set de antrenare / dezvoltare / test (fără scurgeri între înregistrări)
    validate     4. validare manuală 10% din test
    dataset      5. vocabular, procesor, Dataset PyTorch, collator CTC
    train        6. antrenare (fine-tuning Wav2Vec2-CTC)
    evaluate     7. evaluare (WER, CER, tipuri de erori)
    inference    * inferență (folosită de server.py)
"""

__version__ = "1.1.0"

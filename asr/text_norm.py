"""Normalizarea textului pentru ASR în limba română.

Vocabularul modelului: a-z, ă â î ș ț și spațiu. Tot ce nu e în vocabular
este convertit (cifre -> litere, simboluri matematice -> cuvinte) sau eliminat.
Acest pas contează mult pentru lecțiile de matematică, unde apar numere și
simboluri ("12 + 8 = 20").
"""
from __future__ import annotations

import re
import unicodedata

RO_LETTERS = "abcdefghijklmnopqrstuvwxyzăâîșț"
VOCAB_CHARS = list(RO_LETTERS) + [" "]
_ALLOWED = set(VOCAB_CHARS)

# cedilă (forma veche, ş ţ) -> virgulă (forma corectă, ș ț)
_DIACRITIC_FIX = str.maketrans({"ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț"})

_SYMBOLS_RO = {
    "+": " plus ",
    "=": " egal ",
    "×": " ori ",
    "·": " ori ",
    "*": " ori ",
    "÷": " împărțit la ",
    "%": " la sută ",
    "<": " mai mic decât ",
    ">": " mai mare decât ",
    "°": " grade ",
    "€": " euro ",
    "$": " dolari ",
}

_NUM_RE = re.compile(r"\d+(?:[ .]\d{3})*(?:,\d+)?")


def _num_to_words(num: str, lang: str = "ro") -> str:
    try:
        from num2words import num2words
    except ImportError:  # num2words lipsește -> eliminăm cifrele în loc să crăpăm
        return " "
    s = num.replace(" ", "").replace(".", "")
    try:
        if "," in s:
            whole, frac = s.split(",", 1)
            return (f"{num2words(int(whole), lang=lang)} virgulă "
                    f"{' '.join(num2words(int(c), lang=lang) for c in frac)}")
        return num2words(int(s), lang=lang)
    except Exception:
        return " "


def normalize_text(text: str, lang: str = "ro") -> str:
    """Normalizează o transcriere: lowercase, diacritice, cifre -> litere, curățare."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).translate(_DIACRITIC_FIX)
    text = text.replace("ã", "ă").replace("Ã", "Ă")
    text = text.lower()
    if lang == "ro":
        for sym, word in _SYMBOLS_RO.items():
            text = text.replace(sym, word)
        text = _NUM_RE.sub(lambda m: f" {_num_to_words(m.group(0), lang)} ", text)
    out = []
    for ch in text:
        if ch in _ALLOWED:
            out.append(ch)
        elif ch.isalpha():
            # litere străine (é, ö, ...) -> litera de bază; orice altceva -> spațiu
            base = unicodedata.normalize("NFKD", ch)
            base = "".join(c for c in base if not unicodedata.combining(c))
            out.append(base if base and all(c in _ALLOWED for c in base) else " ")
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_ru(text: str) -> str:
    """Normalizare minimă pentru rusă: lowercase, ё->е, fără punctuație."""
    text = unicodedata.normalize("NFC", text or "").lower().replace("ё", "е")
    text = re.sub(r"[^а-я ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

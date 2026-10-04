"""Ultra-fast deterministic language detection for en, es, fr, de, ja.

Executes in < 2ms without heavy external library dependencies or network calls.
"""
import re
from typing import Tuple


def detect_language(text: str) -> str:
    """Detects primary language: 'en', 'es', 'fr', 'de', or 'ja'."""
    text_clean = text.strip()
    if not text_clean:
        return "en"

    # 1. Japanese check (Hiragana, Katakana, Kanji)
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text_clean):
        return "ja"

    text_lower = text_clean.lower()

    # 2. German character & stopword cues
    german_chars = re.search(r"[äöüß]", text_lower)
    german_words = {"und", "ist", "ich", "mein", "meine", "bitte", "bestellung", "rueckgabe", "rückgabe", "versand", "garantie", "lieferung", "danke", "nicht", "haben", "kann", "sie", "wir", "der", "die", "das", "ein", "eine"}
    words = set(re.findall(r"\b\w+\b", text_lower))
    if german_chars or len(words.intersection(german_words)) >= 1:
        return "de"

    # 3. Spanish cues (inverted punctuation, diacritics, stopwords)
    if "¿" in text or "¡" in text:
        return "es"
    spanish_chars = re.search(r"[ñáíóú]", text_lower)
    spanish_words = {"hola", "pedido", "devolucion", "devolución", "envio", "envío", "reembolso", "garantia", "garantía", "puedo", "gracias", "donde", "dónde", "mi", "mis", "por", "favor", "quiero", "este", "esta", "para"}
    if spanish_chars or len(words.intersection(spanish_words)) >= 1:
        return "es"

    # 4. French cues (diacritics, contractions, stopwords)
    french_chars = re.search(r"[éèêàçœù]", text_lower)
    french_words = {"bonjour", "commande", "retour", "retourner", "livraison", "remboursement", "est-ce", "merci", "avec", "pour", "suis", "garantie", "je", "voudrais", "mon", "ma", "mes", "votre", "vos", "avoir", "faire", "ou"}
    if french_chars or len(words.intersection(french_words)) >= 1:
        return "fr"

    # Default to English
    return "en"

"""Sentiment and frustration scoring for customer support interactions."""
import re
from typing import Dict, Any

FRUSTRATION_TERMS = {
    "en": {"ridiculous", "terrible", "awful", "scam", "useless", "worst", "unacceptable", "furious", "angry", "lawyer", "refund now", "broken"},
    "es": {"ridículo", "terrible", "pésimo", "estafa", "inútil", "inaceptable", "furioso", "enfadado", "abogado", "denuncia", "vergüenza"},
    "fr": {"ridicule", "terrible", "horrible", "arnaque", "inutile", "inacceptable", "furieux", "en colère", "plainte", "scandale", "honteux"},
    "de": {"lächerlich", "schrecklich", "katastrophe", "betrug", "nutzlos", "unverschämt", "wütend", "anwalt", "abzocke", "verarscht"},
    "ja": {"最悪", "詐欺", "使えない", "ひどい", "酷い", "怒り", "許せない", "ふざけるな", "役立たず", "訴訟", "弁護士"},
}


def compute_sentiment_score(text: str, language: str = "en") -> float:
    """Computes frustration score from 0.0 (calm) to 1.0 (extremely frustrated)."""
    text_clean = text.strip()
    if not text_clean:
        return 0.0

    score = 0.0
    text_lower = text_clean.lower()

    # 1. Frustration keywords check
    terms = FRUSTRATION_TERMS.get(language, FRUSTRATION_TERMS["en"])
    all_terms = set.union(*FRUSTRATION_TERMS.values())
    
    matched_terms = [t for t in all_terms if t in text_lower]
    score += len(matched_terms) * 0.35

    # 2. Aggressive punctuation (!? or multiple exclamation marks)
    if re.search(r"[!?]{2,}", text_clean):
        score += 0.25
    elif "!" in text_clean:
        score += 0.10

    # 3. Capitalization ratio (shouting in Latin scripts)
    letters = [c for c in text_clean if c.isalpha()]
    if len(letters) > 6:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.6:
            score += 0.30

    return min(1.0, round(score, 2))

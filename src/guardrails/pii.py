"""PII (Personally Identifiable Information) masking and redaction for logs and LLM inputs."""
import re
from typing import Dict, Any, Tuple

# Pre-compiled high-efficiency regex patterns (< 2ms)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Masks emails, credit cards, phone numbers, and SSNs.

    Returns:
        (sanitized_text, redaction_counts)
    """
    counts = {"email": 0, "credit_card": 0, "phone": 0, "ssn": 0}

    # Redact Emails
    emails = EMAIL_PATTERN.findall(text)
    if emails:
        counts["email"] = len(emails)
        text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)

    # Redact Credit Cards
    cards = CREDIT_CARD_PATTERN.findall(text)
    if cards:
        counts["credit_card"] = len(cards)
        text = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", text)

    # Redact SSNs
    ssns = SSN_PATTERN.findall(text)
    if ssns:
        counts["ssn"] = len(ssns)
        text = SSN_PATTERN.sub("[REDACTED_SSN]", text)

    # Redact Phone Numbers (minimum 8 digits)
    phones = [p for p in PHONE_PATTERN.findall(text) if sum(c.isdigit() for c in p) >= 8]
    if phones:
        counts["phone"] = len(phones)
        for p in phones:
            text = text.replace(p.strip(), "[REDACTED_PHONE]")

    return text, counts

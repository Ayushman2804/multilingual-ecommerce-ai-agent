"""Intent classification and entity extraction for customer requests."""
import re
from typing import Dict, Any, Optional
from enum import Enum


class Intent(str, Enum):
    FAQ = "faq"
    ORDER_LOOKUP = "order_lookup"
    RETURN_REQUEST = "return_request"
    PRODUCT_INQUIRY = "product_inquiry"
    ESCALATION = "escalation"


ORDER_ID_PATTERN = re.compile(r"\b(ORD-\d{4})\b", re.IGNORECASE)

ESCALATION_KEYWORDS = {
    # English
    "human", "agent", "representative", "manager", "lawyer", "sue", "scam", "fraud", "unacceptable", "furious",
    # Spanish
    "humano", "agente", "representante", "gerente", "estafa", "denuncia", "furioso",
    # French
    "humain", "responsable", "directeur", "arnaque", "escroquerie", "inacceptable",
    # German
    "mensch", "mitarbeiter", "vorgesetzter", "anwalt", "betrug", "unverschämt",
    # Japanese
    "担当者", "オペレーター", "人間", "責任者", "詐欺", "訴訟",
}

RETURN_KEYWORDS = {
    "return", "refund", "devolver", "devolución", "reembolso", "retourner", "retour",
    "remboursement", "rücksendung", "zurückgeben", "erstatten", "返品", "返金",
}

ORDER_STATUS_KEYWORDS = {
    "status", "where is", "track", "tracking", "shipped", "arrived", "delivery",
    "dónde está", "estado", "seguimiento", "où est", "suivi", "statut",
    "wo ist", "status", "sendungsverfolgung", "どこ", "状況", "追跡",
}

PRODUCT_SYNONYMS = {
    "headphones": "Headphones",
    "auriculares": "Headphones",
    "casque": "Headphones",
    "kopfhörer": "Headphones",
    "kopfhoerer": "Headphones",
    "ヘッドホン": "Headphones",
    "watch": "Watch",
    "reloj": "Watch",
    "montre": "Watch",
    "uhr": "Watch",
    "時計": "Watch",
    "keyboard": "Keyboard",
    "teclado": "Keyboard",
    "clavier": "Keyboard",
    "tastatur": "Keyboard",
    "キーボード": "Keyboard",
    "monitor": "Monitor",
    "pantalla": "Monitor",
    "ecran": "Monitor",
    "écran": "Monitor",
    "bildschirm": "Monitor",
    "モニター": "Monitor",
}


def route_intent(query: str) -> Dict[str, Any]:
    """Analyzes intent and extracts relevant entities in < 5ms."""
    q_lower = query.lower()

    # 1. Check for escalation cues
    for kw in ESCALATION_KEYWORDS:
        if kw in q_lower:
            return {"intent": Intent.ESCALATION, "reason": f"Trigger keyword: {kw}"}

    # 2. Extract Order ID if present
    match = ORDER_ID_PATTERN.search(query)
    extracted_order_id = match.group(1).upper() if match else None

    # 3. Check for Return / Refund with Order ID
    if any(kw in q_lower for kw in RETURN_KEYWORDS):
        if extracted_order_id:
            return {
                "intent": Intent.RETURN_REQUEST,
                "order_id": extracted_order_id,
                "reason": query,
            }
        return {"intent": Intent.FAQ, "topic": "returns"}

    # 4. Check for Order Tracking / Status with Order ID
    if extracted_order_id or any(kw in q_lower for kw in ORDER_STATUS_KEYWORDS):
        if extracted_order_id:
            return {"intent": Intent.ORDER_LOOKUP, "order_id": extracted_order_id}
        return {"intent": Intent.ORDER_LOOKUP, "order_id": None}

    # 5. Check for Product Search & Canonical Synonyms
    for syn_kw, canonical in PRODUCT_SYNONYMS.items():
        if syn_kw in q_lower:
            return {"intent": Intent.PRODUCT_INQUIRY, "query": canonical}

    if any(kw in q_lower for kw in ["product", "price", "stock", "buy", "precio", "prix", "preis", "価格", "在庫"]):
        return {"intent": Intent.PRODUCT_INQUIRY, "query": query}

    # Default to Knowledge Base (FAQ)
    return {"intent": Intent.FAQ}

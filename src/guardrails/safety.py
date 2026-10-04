"""Safety guardrails: Prompt injection defense and Out-of-Scope refusals."""
import re
from typing import Dict, Any, Tuple

# Prompt Injection Patterns across languages
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt(\s+override|\s+leak|\s+reveal)?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unrestricted|an\s+AI\s+without\s+rules)", re.IGNORECASE),
    re.compile(r"bypass\s+all\s+(safety|rules|filters)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+all\s+rules", re.IGNORECASE),
    re.compile(r"modo\s+desarrollador|ignora\s+instrucciones", re.IGNORECASE),
    re.compile(r"ignorer?\s+toutes?\s+les?\s+instructions?", re.IGNORECASE),
    re.compile(r"ignoriere\s+alle\s+anweisungen", re.IGNORECASE),
    re.compile(r"プロンプトを(表示|無視|忘れて)", re.IGNORECASE),
]

# Out-of-Scope Topics (non-ecommerce)
OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(write\s+(a\s+)?(python|javascript|c\+\+|code|poem|story|essay))\b", re.IGNORECASE),
    re.compile(r"\b(diagnose\s+my\s+(illness|symptoms|disease)|prescribe|medical\s+advice)\b", re.IGNORECASE),
    re.compile(r"\b(who\s+won\s+the\s+election|political\s+opinion|who\s+should\s+i\s+vote\s+for)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+make\s+a\s+(bomb|weapon|explosive|malware))\b", re.IGNORECASE),
    re.compile(r"\b(écrire\s+(du\s+)?code|rédiger\s+un\s+poème)\b", re.IGNORECASE),
    re.compile(r"\b(escribir\s+(un\s+)?código|poema|consejo\s+médico)\b", re.IGNORECASE),
    re.compile(r"\b(schreibe\s+(ein\s+)?(gedicht|programmcode))\b", re.IGNORECASE),
    re.compile(r"(コードを書いて|詩を書いて|医療診断|政治的意見)", re.IGNORECASE),
]

LOCALIZED_REFUSALS = {
    "injection": {
        "en": "I cannot fulfill requests attempting to alter system instructions. How may I assist you with your order or our store policies?",
        "es": "No puedo procesar solicitudes que intenten modificar las instrucciones del sistema. ¿En qué puedo ayudarle respecto a su pedido o nuestras políticas?",
        "fr": "Je ne peux pas répondre aux demandes visant à modifier les instructions du système. Comment puis-je vous aider concernant votre commande ou nos politiques ?",
        "de": "Ich kann Anfragen, die Systemanweisungen zu umgehen versuchen, nicht bearbeiten. Wie kann ich Ihnen bei Ihrer Bestellung oder unseren Richtlinien helfen?",
        "ja": "システム設定やプロンプトの変更に関するご要望にはお応えできません。ご注文や店舗ポリシーについて何かお手伝いできることはございますか？",
    },
    "out_of_scope": {
        "en": "As an e-commerce support assistant, I can only assist with orders, returns, shipping, warranty, and product catalog inquiries.",
        "es": "Como asistente de soporte de compras, solo puedo ayudarle con pedidos, devoluciones, envíos, garantías y catálogo de productos.",
        "fr": "En tant qu'assistant de service client, je peux uniquement vous aider concernant les commandes, retours, livraisons, garanties et notre catalogue.",
        "de": "Als Kundenservice-Assistent kann ich Ihnen ausschließlich bei Fragen zu Bestellungen, Retouren, Versand, Garantie und unserem Sortiment weiterhelfen.",
        "ja": "当窓口はカスタマーサポート専用です。ご注文、配送、返品、保証、および取扱商品に関するご案内のみを承っております。",
    },
}


def check_safety_and_scope(text: str, language: str = "en") -> Tuple[bool, str, Dict[str, Any]]:
    """Checks input against prompt injections and out-of-scope tasks.

    Returns:
        (is_safe, refusal_message_or_empty, metadata)
    """
    # 1. Prompt Injection Check
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            refusal = LOCALIZED_REFUSALS["injection"].get(language, LOCALIZED_REFUSALS["injection"]["en"])
            return False, refusal, {"violation": "prompt_injection", "pattern": pattern.pattern}

    # 2. Out-of-Scope Check
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            refusal = LOCALIZED_REFUSALS["out_of_scope"].get(language, LOCALIZED_REFUSALS["out_of_scope"]["en"])
            return False, refusal, {"violation": "out_of_scope", "pattern": pattern.pattern}

    return True, "", {}

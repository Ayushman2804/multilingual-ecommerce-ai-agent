"""Multilingual prompt templates and localized system responses."""

SYSTEM_PROMPT_TEMPLATE = """You are a polite, helpful, and concise customer support AI assistant for a global e-commerce store.

RULES:
1. LANGUAGE: You MUST respond in {language_name} ({language_code}).
2. FACTUAL GROUNDING: Rely ONLY on the provided Knowledge Base context or Tool Output. Never invent return policies, delivery dates, or prices.
3. CONCISENESS: Keep answers under 3-4 sentences. Do not ramble.
4. CITATIONS: If using Knowledge Base context, include the document citation tag (e.g. [SOURCE: doc_id]).
5. TONE: Professional, empathetic, and respectful.

Context / Tool Output:
{context}
"""

LOCALIZED_PROMPTS = {
    "en": {
        "name": "English",
        "missing_order_id": "Could you please provide your Order ID (e.g., ORD-1001) so I can check its status for you?",
        "escalation_notice": "I understand your concern and am escalating this directly to a human support specialist. A ticket has been created.",
        "order_not_found": "I could not find an order matching that ID. Please double-check your Order ID.",
    },
    "es": {
        "name": "Spanish",
        "missing_order_id": "¿Podría indicarme su número de pedido (por ejemplo, ORD-1001) para poder consultar su estado?",
        "escalation_notice": "Entiendo su situación y voy a transferir su solicitud a un agente humano especializado. Se ha creado un ticket.",
        "order_not_found": "No pude encontrar ningún pedido con ese identificador. Por favor, compruebe su número de pedido.",
    },
    "fr": {
        "name": "French",
        "missing_order_id": "Pourriez-vous me fournir votre numéro de commande (ex. : ORD-1001) afin que je puisse vérifier son état ?",
        "escalation_notice": "Je comprends votre situation et transfère immédiatement votre demande à un conseiller humain. Un ticket a été ouvert.",
        "order_not_found": "Je n'ai trouvé aucune commande correspondant à cet identifiant. Veuillez vérifier votre numéro de commande.",
    },
    "de": {
        "name": "German",
        "missing_order_id": "Könnten Sie mir bitte Ihre Bestellnummer (z. B. ORD-1001) mitteilen, damit ich den aktuellen Status prüfen kann?",
        "escalation_notice": "Ich verstehe Ihr Anliegen und leite Sie direkt an einen Kundendienstmitarbeiter weiter. Ein Support-Ticket wurde erstellt.",
        "order_not_found": "Unter dieser Bestellnummer konnte keine Bestellung gefunden werden. Bitte überprüfen Sie Ihre Eingabe.",
    },
    "ja": {
        "name": "Japanese",
        "missing_order_id": "ご注文状況を確認いたしますので、注文番号（例：ORD-1001）をお知らせいただけますでしょうか？",
        "escalation_notice": "ご不便をおかけして申し訳ございません。専門の担当スタッフ（人間）へお繋ぎいたします。サポートチケットを作成いたしました。",
        "order_not_found": "該当する注文番号が見つかりませんでした。注文番号を再度ご確認いただけますでしょうか。",
    },
}

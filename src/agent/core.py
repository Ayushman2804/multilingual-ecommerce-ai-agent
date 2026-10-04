"""Core Multilingual Agent coordinator.

Performs deterministic sub-5ms intent & language routing, executes tools or
retrieves knowledge base articles, and synthesizes localized responses.
"""
import time
from typing import Dict, Any, Optional

from src.agent.lang_detector import detect_language
from src.agent.router import route_intent, Intent
from src.agent.prompts import LOCALIZED_PROMPTS
from src.tools.order_tools import lookup_order, request_return_or_refund, lookup_product
from src.rag.retriever import MultilingualRetriever


class MultilingualCSAgent:
    def __init__(self, retriever: MultilingualRetriever):
        self.retriever = retriever

    def process_message(self, message: str, user_language: Optional[str] = None) -> Dict[str, Any]:
        """Processes incoming user query, routes to tool or RAG, and produces response."""
        start_time = time.perf_counter()

        # 1. Detect language (sub-2ms)
        lang = user_language or detect_language(message)
        prompts = LOCALIZED_PROMPTS.get(lang, LOCALIZED_PROMPTS["en"])

        # 2. Intent routing (sub-5ms)
        route_info = route_intent(message)
        intent = route_info["intent"]

        response_text = ""
        tool_data = None
        citations = []
        is_escalated = False

        # 3. Execution Branching
        if intent == Intent.ESCALATION:
            is_escalated = True
            response_text = prompts["escalation_notice"]

        elif intent == Intent.ORDER_LOOKUP:
            order_id = route_info.get("order_id")
            if not order_id:
                response_text = prompts["missing_order_id"]
            else:
                order = lookup_order(order_id)
                tool_data = order
                if not order["found"]:
                    response_text = prompts["order_not_found"]
                else:
                    response_text = self._format_order_response(order, lang)

        elif intent == Intent.RETURN_REQUEST:
            order_id = route_info.get("order_id")
            if not order_id:
                response_text = prompts["missing_order_id"]
            else:
                refund_res = request_return_or_refund(order_id, reason=message)
                tool_data = refund_res
                if refund_res["success"]:
                    response_text = self._format_refund_success(refund_res, lang)
                else:
                    response_text = self._format_refund_error(refund_res, lang)

        elif intent == Intent.PRODUCT_INQUIRY:
            catalog_res = lookup_product(route_info["query"])
            tool_data = catalog_res
            response_text = self._format_catalog_response(catalog_res, lang)

        else:  # Intent.FAQ (RAG retrieval)
            rag_res = self.retriever.retrieve(message, language=lang, top_k=2)
            citations = [doc["id"] for doc in rag_res["results"]]
            if rag_res["results"]:
                top_doc = rag_res["results"][0]
                response_text = f"{top_doc['content']} [SOURCE: {top_doc['id']}]"
            else:
                response_text = "I could not find a relevant policy. Let me connect you with an agent."

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": message,
            "language": lang,
            "intent": intent.value,
            "response": response_text,
            "tool_data": tool_data,
            "citations": citations,
            "is_escalated": is_escalated,
            "latency_ms": round(elapsed_ms, 2),
        }

    def _format_order_response(self, order: Dict[str, Any], lang: str) -> str:
        status = order['status']
        order_id = order['order_id']
        item = order['item_name']
        tracking = order.get('tracking_number') or 'N/A'

        templates = {
            "en": f"Your order {order_id} ({item}) is currently **{status}**. Tracking number: {tracking}.",
            "es": f"Su pedido {order_id} ({item}) está actualmente **{status}**. Número de seguimiento: {tracking}.",
            "fr": f"Votre commande {order_id} ({item}) est actuellement **{status}**. Numéro de suivi : {tracking}.",
            "de": f"Ihre Bestellung {order_id} ({item}) befindet sich im Status **{status}**. Sendungsverfolgungsnummer: {tracking}.",
            "ja": f"お客様のご注文 {order_id}（{item}）の現在のステータスは「**{status}**」です。追跡番号：{tracking}。",
        }
        return templates.get(lang, templates["en"])

    def _format_refund_success(self, refund: Dict[str, Any], lang: str) -> str:
        refund_id = refund["refund_id"]
        amt = refund["amount"]
        curr = refund["currency"]

        templates = {
            "en": f"Return approved! Refund reference: {refund_id} for {amt} {curr}. A prepaid shipping label has been emailed to you.",
            "es": f"¡Devolución aprobada! Referencia: {refund_id} por {amt} {curr}. Le hemos enviado por correo la etiqueta prepagada.",
            "fr": f"Retour validé ! Référence de remboursement : {refund_id} d'un montant de {amt} {curr}. L'étiquette de retour vous a été envoyée par email.",
            "de": f"Rückgabe genehmigt! Erstattungsreferenz: {refund_id} über {amt} {curr}. Ein frankiertes Rücksendeetikett wurde per E-Mail versandt.",
            "ja": f"返品申請が承認されました。返金参照番号：{refund_id}（金額：{amt} {curr}）。返送用ラベルをメールにてお送りいたしました。",
        }
        return templates.get(lang, templates["en"])

    def _format_refund_error(self, refund: Dict[str, Any], lang: str) -> str:
        err = refund.get("error", "Return not eligible.")
        templates = {
            "en": f"Unable to process return: {err}",
            "es": f"No se pudo tramitar la devolución: {err}",
            "fr": f"Impossible de traiter le retour : {err}",
            "de": f"Rückgabe konnte nicht verarbeitet werden: {err}",
            "ja": f"返品処理を完了できませんでした：{err}",
        }
        return templates.get(lang, templates["en"])

    def _format_catalog_response(self, catalog: Dict[str, Any], lang: str) -> str:
        items = catalog.get("products", [])
        if not items:
            templates = {
                "en": "We currently do not have items matching that description in stock.",
                "es": "Actualmente no tenemos productos en stock que coincidan con esa descripción.",
                "fr": "Nous n'avons actuellement aucun article correspondant en stock.",
                "de": "Derzeit haben wir keine passenden Artikel auf Lager.",
                "ja": "申し訳ございません。現在該当する商品の在庫がございません。",
            }
            return templates.get(lang, templates["en"])

        top = items[0]
        name, price, curr = top["name"], top["price"], top["currency"]
        templates = {
            "en": f"Found: {name} at {price} {curr} ({top['description']}).",
            "es": f"Encontrado: {name} por {price} {curr} ({top['description']}).",
            "fr": f"Trouvé : {name} à {price} {curr} ({top['description']}).",
            "de": f"Gefunden: {name} für {price} {curr} ({top['description']}).",
            "ja": f"商品が見つかりました：{name}（価格：{price} {curr}、{top['description']}）。",
        }
        return templates.get(lang, templates["en"])

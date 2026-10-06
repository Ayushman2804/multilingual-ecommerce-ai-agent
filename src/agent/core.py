import time
import uuid
from typing import Dict, Any, Optional

from src.agent.lang_detector import detect_language
from src.agent.router import route_intent, Intent
from src.agent.prompts import LOCALIZED_PROMPTS
from src.agent.memory import session_manager, ConversationSession
from src.agent.sentiment import compute_sentiment_score
from src.agent.escalation import generate_handoff_ticket, HandoffTicket
from src.agent.llm import llm_synthesizer
from src.guardrails.pii import mask_pii
from src.guardrails.safety import check_safety_and_scope
from src.tools.order_tools import lookup_order, request_return_or_refund, lookup_product
from src.rag.retriever import MultilingualRetriever


class MultilingualCSAgent:
    def __init__(self, retriever: MultilingualRetriever):
        self.retriever = retriever

    def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes incoming user query, applies guardrails, updates memory, and synthesizes response."""
        start_time = time.perf_counter()

        # 0. PII Redaction Guardrail
        sanitized_message, pii_counts = mask_pii(message)

        # 1. Session Memory setup
        sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        session = session_manager.get_or_create(sess_id)

        # 2. Language Detection
        lang = user_language or detect_language(sanitized_message)
        session.user_language = lang
        prompts = LOCALIZED_PROMPTS.get(lang, LOCALIZED_PROMPTS["en"])

        # 3. Prompt-Injection & Scope Guardrail Check
        is_safe, refusal_msg, safety_meta = check_safety_and_scope(sanitized_message, lang)
        if not is_safe:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            session.add_turn(role="user", content=sanitized_message, language=lang, intent="blocked")
            session.add_turn(role="assistant", content=refusal_msg, language=lang, intent="blocked")
            return {
                "session_id": session.session_id,
                "query": sanitized_message,
                "language": lang,
                "intent": "blocked_by_guardrails",
                "response": refusal_msg,
                "tool_data": None,
                "citations": [],
                "is_escalated": False,
                "frustration_score": session.frustration_score,
                "handoff_ticket": None,
                "pii_redacted": pii_counts,
                "safety_metadata": safety_meta,
                "latency_ms": round(elapsed_ms, 2),
            }

        # 4. Sentiment analysis & frustration tracking
        turn_sentiment = compute_sentiment_score(sanitized_message, lang)
        session.frustration_score = max(session.frustration_score, turn_sentiment)

        # 5. Intent routing (sub-5ms)
        route_info = route_intent(sanitized_message)
        intent = route_info["intent"]

        # Track referenced orders
        if route_info.get("order_id"):
            session.order_ids_referenced.append(route_info["order_id"])

        response_text = ""
        tool_data = None
        citations = []
        is_escalated = False
        handoff_ticket: Optional[HandoffTicket] = None

        # Check frustration threshold (> 0.60)
        if session.frustration_score >= 0.60 and intent != Intent.ESCALATION:
            intent = Intent.ESCALATION
            route_info["reason"] = f"High customer frustration detected ({session.frustration_score:.2f})"

        # 4. Execution Branching
        if intent == Intent.ESCALATION:
            is_escalated = True
            session.is_escalated = True
            reason = route_info.get("reason", "Customer requested human representative.")
            session.escalation_reason = reason
            response_text = prompts["escalation_notice"]
            handoff_ticket = generate_handoff_ticket(session, reason)

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
                fallback = f"{top_doc['content']} [SOURCE: {top_doc['id']}]"
                response_text = llm_synthesizer.generate_response(
                    query=sanitized_message,
                    language=lang,
                    context=rag_res["context_prompt"],
                    default_fallback=fallback,
                )
            else:
                response_text = "I could not find a relevant policy. Let me connect you with an agent."

        # If LLM is active and this was a tool or catalog query, synthesize a thoughtful response
        if llm_synthesizer.is_active and intent in [Intent.ORDER_LOOKUP, Intent.RETURN_REQUEST, Intent.PRODUCT_INQUIRY] and tool_data:
            response_text = llm_synthesizer.generate_response(
                query=sanitized_message,
                language=lang,
                tool_data=tool_data,
                default_fallback=response_text,
            )

        # 5. Record session history
        session.add_turn(role="user", content=message, language=lang, intent=intent.value)
        session.add_turn(role="assistant", content=response_text, language=lang, intent=intent.value)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "session_id": session.session_id,
            "query": sanitized_message,
            "language": lang,
            "intent": intent.value,
            "response": response_text,
            "tool_data": tool_data,
            "citations": citations,
            "is_escalated": is_escalated,
            "frustration_score": session.frustration_score,
            "handoff_ticket": handoff_ticket.model_dump() if handoff_ticket else None,
            "pii_redacted": pii_counts,
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

"""LLM Synthesis Engine for thoughtful, multilingual, and low-latency customer responses.

Supports:
1. Groq (Free Tier, Llama 3.3 70B / Llama 3.1 8B @ ~300-500 tok/sec)
2. Google Gemini Flash (Free Tier via OpenAI compatibility endpoint)
3. Graceful Local Fallback (Deterministic synthesized response)
"""
import os
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Generator
from openai import OpenAI

logger = logging.getLogger(__name__)

THOUGHTFUL_SYSTEM_PROMPT = """You are an empathetic, highly intelligent, and concise senior customer support specialist for an international e-commerce store.

YOUR OBJECTIVES:
1. LANGUAGE CONSISTENCY: You MUST reply fluently and naturally in {language_name} ({language_code}).
2. THOUGHTFUL RESOLUTION:
   - Acknowledge the customer's concern with genuine empathy.
   - Address their exact issue directly using the provided Context and Tool Data.
   - Provide clear, actionable next steps (e.g. tracking steps, return label instructions, delivery timelines).
3. FACTUAL GROUNDING:
   - Never invent order details, tracking statuses, delivery dates, or return policies.
   - Rely strictly on the supplied Context / Tool Data.
   - If citing store policies, include the citation tag (e.g. [SOURCE: doc_id]).
4. CONCISENESS: Keep the final response within 2-4 clear sentences so the customer can read it quickly.

Language: {language_name} ({language_code})
Context / Policy Knowledge Base:
{context}

Tool Execution Data:
{tool_data}
"""

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
}


class LLMSynthesizer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client: Optional[OpenAI] = None
        self.model_name = "llama-3.3-70b-versatile"

        # 1. Initialize Groq (Free high-speed tier)
        if self.groq_api_key:
            self.client = OpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            self.model_name = "llama-3.3-70b-versatile"
            logger.info("LLMSynthesizer initialized with Groq Free Tier (Llama 3.3 70B).")
        # 2. Initialize Gemini (Free Google AI Studio tier)
        elif self.gemini_api_key:
            self.client = OpenAI(
                api_key=self.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.model_name = "gemini-1.5-flash"
            logger.info("LLMSynthesizer initialized with Gemini 1.5 Flash Free Tier.")
        else:
            logger.info("No external LLM API key detected. Using local synthesis fallback.")

    @property
    def is_active(self) -> bool:
        return self.client is not None

    def generate_response(
        self,
        query: str,
        language: str,
        context: str = "",
        tool_data: Optional[Dict[str, Any]] = None,
        default_fallback: str = "",
    ) -> str:
        """Generates a thoughtful grounded response via free LLM or fallback."""
        if not self.is_active:
            return default_fallback

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system_prompt = THOUGHTFUL_SYSTEM_PROMPT.format(
            language_name=lang_name,
            language_code=language,
            context=context or "No relevant policy documents found.",
            tool_data=json.dumps(tool_data, ensure_ascii=False) if tool_data else "No tool data.",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            return content.strip() if content else default_fallback
        except Exception as e:
            logger.warning(f"External LLM invocation failed ({e}). Falling back to local template.")
            return default_fallback

    def stream_response(
        self,
        query: str,
        language: str,
        context: str = "",
        tool_data: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Yields response tokens in real-time for perceived TTFT < 200ms."""
        if not self.is_active:
            return

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system_prompt = THOUGHTFUL_SYSTEM_PROMPT.format(
            language_name=lang_name,
            language_code=language,
            context=context or "No relevant policy documents found.",
            tool_data=json.dumps(tool_data, ensure_ascii=False) if tool_data else "No tool data.",
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=300,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"LLM streaming failed ({e}).")


llm_synthesizer = LLMSynthesizer()

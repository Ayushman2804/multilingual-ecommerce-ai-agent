# Multilingual E-Commerce AI Customer Service Agent

A production-grade, low-latency multilingual customer service AI agent built for e-commerce platforms. Supports English, Spanish, French, German, and Japanese with sub-3-second end-to-end response times, RAG knowledge retrieval, real-time tool calling, memory management, and human handoff.

## Architecture Highlights
- **Target Response Time**: < 3.0s total wall time (< 500ms TTFT via Server-Sent Events).
- **Core Stack**: Python 3.10+, FastAPI, LangChain, FAISS, Docker, GCP Cloud Run / vLLM.
- **Multilingual Support**: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Japanese (`ja`).
- **Knowledge Base (RAG)**: In-process FAISS `IndexFlatIP` vector index with normalized embeddings via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Tool Calling**: Mock order lookup, return processing, and catalog queries backed by an indexed SQLite store.

## Latency Budget Breakdown (< 3000ms SLA)
- Pre-LLM Overhead (PII Redaction, Language Detection, FAISS Search, Tool Execution): ~75–100ms
- Perceived Time-to-First-Token (TTFT): ~250–350ms
- Generation Time (120–150 tokens): ~1200–2500ms (vLLM / fast hosted fallback)

## Project Roadmap
- [x] **Phase 1**: Architecture, latency budget analysis, and design doc.
- [x] **Phase 2**: Knowledge base and multilingual RAG with source citations.
- [x] **Phase 3**: Agent logic, language detection, and mock tool calling.
- [ ] **Phase 4**: Session memory, sentiment analysis, and human escalation.
- [ ] **Phase 5**: Guardrails, prompt injection defense, and PII masking.
- [ ] **Phase 6**: FastAPI backend with SSE streaming and chat UI.
- [ ] **Phase 7**: Multilingual evaluation suite across 50+ queries.
- [ ] **Phase 8**: Containerization (Docker) and GCP deployment configs.
- [ ] **Phase 9**: Portfolio packaging and benchmark documentation.

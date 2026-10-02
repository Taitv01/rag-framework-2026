# RAG Upgrade Roadmap 2026

This roadmap summarizes the main upgrade pillars for moving the framework from
basic RAG to advanced and agentic RAG workflows.

## 1. Graph RAG

- Build and query knowledge graphs with NetworkX and optional Neo4j storage.
- Extract entities and relations from documents.
- Support global relationship questions that are hard for chunk-only retrieval.

## 2. Hierarchical Retrieval

- Use parent-child retrieval: small child chunks for matching, larger parent
  chunks for answer context.
- Add metadata enrichment for characters, locations, time periods, topics, and
  sentiment.
- Preserve long-context story structure for narrative workflows.

## 3. Hybrid Search and Reranking

- Combine dense vector search with BM25 keyword retrieval.
- Use Vietnamese-aware tokenization for sparse search.
- Apply cross-encoder reranking to reduce irrelevant context and lost-in-the-middle failures.

## 4. Agentic and Corrective RAG

- Rewrite weak user queries before retrieval.
- Grade retrieved documents for relevance.
- Fall back to web search only when local retrieval is insufficient and the
  feature is explicitly enabled.
- Grade generated answers for grounding against retrieved context.

## 5. Vietnamese NLP Tuning

- Normalize Unicode and Vietnamese whitespace consistently.
- Use Vietnamese word segmentation for BM25 and metadata extraction.
- Prefer multilingual or Vietnamese embedding/reranking models such as BGE-M3
  and AITeamVN rerankers.

## Current Status

- Phase 1, Phase 2, and Phase 2.5 are implemented.
- Phase 3's deterministic fairy-tale foundation is implemented with
  `CrossStoryRAG`, `FairyTaleDatasetBuilder`, Smart Library, and Vietnamese OCR.
- Phase 4 streaming SSE, Langfuse tracing, Docker Compose, API authentication,
  rate limiting, and deployment documentation are implemented.
- The current hardening pass adds structured, fail-closed hallucination grading,
  true N-hop subgraph extraction, repaired NaiveRAG embedding/retrieval/Markdown
  refresh contracts, import-safe Windows console handling, centralized version
  metadata, and installable API/UI dependency extras.
- Phase 5 multimodal retrieval, richer LangGraph workflows, bilingual output,
  fine-tuning, and the interactive storytelling application remain planned.

## Next Upgrade Plan

The October 2026 code audit and the staged upgrade plan (stabilize, measure,
fix retrieval core, unify generation, production API) live in
[`upgrade_plan_2026.md`](upgrade_plan_2026.md). Stage 0 (stabilization) is done.

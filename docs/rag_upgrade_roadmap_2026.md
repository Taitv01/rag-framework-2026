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
- Phase 3 has started with deterministic fairy tale utilities:
  `CrossStoryRAG` and `FairyTaleDatasetBuilder`.
- Phase 4 should focus on production hardening, CI quality gates, secrets
  hygiene, and deployment docs.

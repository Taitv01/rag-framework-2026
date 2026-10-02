# Architecture Overview

## Overview

Ultimate RAG Framework provides multiple RAG (Retrieval-Augmented Generation) patterns for different use cases. The architecture is modular and extensible, allowing you to choose the right pattern for your needs.

## RAG Patterns

### 1. Naive RAG

The simplest RAG pattern, suitable for basic Q&A systems.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document   │────▶│   Vector    │────▶│     LLM     │
│    Loader    │     │    Store    │     │  Generator  │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Components:**
- Document Loader: Load and parse documents
- Text Splitter: Chunk documents into smaller pieces
- Embeddings: Convert text to vectors
- Vector Store: Store and retrieve vectors
- LLM: Generate answers from context

**Use Cases:**
- Basic Q&A systems
- Document search
- Simple chatbots

### 2. Advanced RAG

Enhanced RAG with hybrid search and re-ranking.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Query     │────▶│   Hybrid    │────▶│  Re-ranker  │
│  Transform   │     │   Search    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Vector    │     │   Document  │
                    │   Search    │     │   Grader    │
                    └─────────────┘     └─────────────┘
```

**Components:**
- Query Transformer: Optimize queries for better retrieval
- Hybrid Search: Combine vector and keyword search
- Re-ranker: Cross-encoder for precision
- Document Grader: Filter irrelevant documents

**Use Cases:**
- Technical documentation
- Research papers
- Complex Q&A

### 3. Agentic RAG

Agent-based RAG with intelligent retrieval decisions.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Agent     │────▶│  Retrieval  │────▶│   Grader    │
│  (LangGraph) │     │   Decision  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                   │                    │
       │                   ▼                    ▼
       │             ┌─────────────┐     ┌─────────────┐
       └─────────────│   Query     │◀────│   Answer    │
                     │  Rewriter   │     │  Generator  │
                     └─────────────┘     └─────────────┘
```

**Components:**
- Agent: Decide whether to retrieve
- Retrieval Decision: Smart routing
- Document Grader: Relevance assessment
- Query Rewriter: Optimize failed queries
- Answer Generator: Generate from context

**Use Cases:**
- Conversational AI
- Complex reasoning
- Multi-step tasks

### 4. Graph RAG

Knowledge graph integration for structured reasoning.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document   │────▶│  Knowledge  │────▶│   Graph     │
│   Parser     │     │   Graph     │     │  Retriever  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Entity    │     │   LLM       │
                    │  Extraction │     │  Generator  │
                    └─────────────┘     └─────────────┘
```

**Components:**
- Document Parser: Extract entities and relationships
- Knowledge Graph: Store structured knowledge
- Entity Extraction: NLP-based extraction
- Graph Retriever: Path-based retrieval

**Use Cases:**
- Enterprise knowledge bases
- Complex reasoning
- Multi-hop questions

### 5. Multimodal RAG

Image/video understanding through an OpenAI-compatible multimodal model such as
Ox Alpha, optionally grounded with text retrieved from the knowledge base.

```
┌─────────────┐                         ┌─────────────┐
│ Image/Video │────────────────────────▶│ Ox Alpha   │
│ URL/Upload  │                         │ Multimodal │
└─────────────┘                         └──────▲──────┘
                                               │
┌─────────────┐     ┌─────────────┐             │
│ Text Query  │────▶│ RAG         │─────────────┘
│             │     │ Retrieval   │  context + citations
└─────────────┘     └─────────────┘
```

**Components:**
- Upload/URL validation for supported image and video formats
- OpenRouter-compatible `image_url` and `video_url` content blocks
- Optional Naive/Advanced text retrieval with stable citations
- Dedicated Ox client isolated from the default text RAG model
- API endpoints: `/ox/chat`, `/ox/analyze`, `/ox/analyze/url`, and `/ox/status`
- Provider-neutral aliases: `/query/multimodal` and `/query/multimodal/url`

**Use Cases:**
- Chart and screenshot analysis grounded in reports
- Video summarization enriched with knowledge-base context
- Visual document understanding

## Core Components

### Document Loader

Handles multiple file formats:
- PDF (pypdf)
- DOCX (python-docx)
- HTML (BeautifulSoup)
- Markdown
- CSV
- JSON

### Text Splitter

Multiple chunking strategies:
- Recursive character splitting (default)
- Sentence-aware splitting
- Semantic splitting (using embeddings)
- Custom delimiter splitting

### Embeddings

Multi-provider support:
- HuggingFace (local)
- OpenAI
- Cohere

### Vector Store

Multiple backends:
- FAISS (in-memory, prototyping)
- ChromaDB (persistent, local)
- Qdrant (production, scalable)

### LLM

Multi-provider support:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (Ollama)

## Configuration

The framework uses a centralized configuration system:

```python
from src.utils.config import Config

config = Config()

# Get LLM config
llm_config = config.get_llm_config()

# Get RAG config
rag_config = config.get_rag_config()
```

## Evaluation

Built-in evaluation metrics:
- Faithfulness: Is the answer grounded in context?
- Answer Relevance: Does the answer address the question?
- Context Precision: Are retrieved documents relevant?
- Context Recall: Are all relevant documents retrieved?

## Extension Points

The framework is designed to be extensible:

1. **Custom Document Loaders**: Add support for new file formats
2. **Custom Embeddings**: Integrate new embedding models
3. **Custom Vector Stores**: Add new vector databases
4. **Custom Retrievers**: Implement new retrieval strategies
5. **Custom Agents**: Create specialized agents

## Performance Considerations

### Caching

- Query caching for repeated questions
- Embedding caching for document chunks
- Response caching for LLM calls

### Optimization

- Batch processing for documents
- Async support for I/O operations
- Lazy loading for large datasets

### Scaling

- Distributed vector stores (Qdrant, Weaviate)
- Load balancing for LLM calls
- Horizontal scaling for document processing

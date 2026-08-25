"""
Advanced RAG Techniques
=======================

Collection of advanced RAG techniques inspired by top repositories.

Techniques included:
- Self-RAG: Self-reflective RAG
- Corrective RAG (CRAG): Dynamic correction
- HyDE: Hypothetical Document Embedding
- HyPE: Hypothetical Prompt Embedding
- Contextual Compression
- Semantic Chunking
- Document Augmentation

Usage:
    from src.rag.advanced_techniques import SelfRAG, CorrectiveRAG, HyDE

    # Self-RAG
    rag = SelfRAG(llm=llm, retriever=retriever)
    answer = rag.query("What is Python?")

    # HyDE
    hyde = HyDE(llm=llm, embeddings=embeddings)
    results = hyde.search("What is Python?", documents)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json

import numpy as np
from langchain_core.documents import Document


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    sources: List[Document]
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


class SelfRAG:
    """
    Self-Reflective RAG.

    Combines retrieval-based and generation-based methods,
    adaptively deciding whether to use retrieved information.

    Process:
    1. Decide if retrieval is needed
    2. Retrieve documents if needed
    3. Evaluate relevance of each document
    4. Generate response
    5. Assess support from documents
    6. Evaluate utility of response

    Reference: "Self-RAG: Learning to Retrieve, Generate, and Critique
    through Self-Reflection" (Asai et al., 2023)

    Example:
        rag = SelfRAG(llm=llm, retriever=retriever)
        answer = rag.query("What is Python?")
    """

    def __init__(self, llm, retriever, max_retries: int = 2):
        """
        Initialize Self-RAG.

        Args:
            llm: LLM instance
            retriever: Retriever instance
            max_retries: Maximum retries for improvement
        """
        self.llm = llm
        self.retriever = retriever
        self.max_retries = max_retries

    def query(self, question: str, k: int = 5) -> RAGResponse:
        """
        Query with self-reflection.

        Args:
            question: Question to ask
            k: Number of documents to retrieve

        Returns:
            RAGResponse with answer and metadata
        """
        # Step 1: Decide if retrieval is needed
        needs_retrieval = self._should_retrieve(question)

        if not needs_retrieval:
            # Direct generation without retrieval
            answer = self._generate_direct(question)
            return RAGResponse(
                answer=answer,
                sources=[],
                confidence=0.7,
                metadata={"retrieval": False}
            )

        # Step 2: Retrieve documents
        docs = self.retriever.search(question, k=k)

        # Step 3: Evaluate relevance
        relevant_docs = self._evaluate_relevance(question, docs)

        # Step 4: Generate response
        answer = self._generate_with_docs(question, relevant_docs)

        # Step 5: Assess support
        is_supported = self._assess_support(question, answer, relevant_docs)

        # Step 6: Improve if not supported
        if not is_supported:
            for _ in range(self.max_retries):
                # Retrieve more documents
                more_docs = self.retriever.search(question, k=k * 2)
                relevant_docs = self._evaluate_relevance(question, more_docs)
                answer = self._generate_with_docs(question, relevant_docs)
                is_supported = self._assess_support(question, answer, relevant_docs)

                if is_supported:
                    break

        # Step 7: Evaluate utility
        utility = self._evaluate_utility(question, answer)

        return RAGResponse(
            answer=answer,
            sources=relevant_docs,
            confidence=utility,
            metadata={
                "retrieval": True,
                "is_supported": is_supported,
                "utility": utility,
            }
        )

    def _should_retrieve(self, question: str) -> bool:
        """Decide if retrieval is needed."""
        prompt = f"""Decide if this question requires retrieving external documents.

Question: {question}

Does this question need external information to answer accurately?
Reply with only 'yes' or 'no'."""

        response = self.llm.generate(prompt).strip().lower()
        return response == "yes"

    def _evaluate_relevance(self, question: str, docs: List[Document]) -> List[Document]:
        """Evaluate document relevance."""
        relevant = []

        for doc in docs:
            prompt = f"""Evaluate if this document is relevant to the question.

Question: {question}

Document:
{doc.page_content[:500]}

Is this document relevant? Reply with only 'yes' or 'no'."""

            response = self.llm.generate(prompt).strip().lower()
            if response == "yes":
                relevant.append(doc)

        return relevant if relevant else docs[:1]

    def _generate_with_docs(self, question: str, docs: List[Document]) -> str:
        """Generate answer using documents."""
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        return self.llm.generate(prompt)

    def _generate_direct(self, question: str) -> str:
        """Generate answer without retrieval."""
        prompt = f"""Answer the following question based on your knowledge.

Question: {question}

Answer:"""

        return self.llm.generate(prompt)

    def _assess_support(self, question: str, answer: str, docs: List[Document]) -> bool:
        """Assess if answer is supported by documents."""
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Is the following answer supported by the provided context?

Context:
{context}

Answer: {answer}

Reply with only 'yes' or 'no'."""

        response = self.llm.generate(prompt).strip().lower()
        return response == "yes"

    def _evaluate_utility(self, question: str, answer: str) -> float:
        """Evaluate utility of the answer."""
        prompt = f"""Rate the utility of this answer on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

Consider:
- Does it directly answer the question?
- Is it informative and complete?
- Is it accurate?

Return ONLY the numeric score."""

        try:
            return float(self.llm.generate(prompt).strip())
        except (ValueError, TypeError):
            return 0.5


class CorrectiveRAG:
    """
    Corrective RAG (CRAG).

    Dynamically evaluates and corrects the retrieval process,
    combining vector databases, web search, and language models.

    Process:
    1. Retrieve documents
    2. Evaluate retrieval quality
    3. If quality is low, rewrite query and search web
    4. Refine knowledge
    5. Generate corrected answer

    Reference: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

    Example:
        rag = CorrectiveRAG(llm=llm, retriever=retriever)
        answer = rag.query("What is Python?")
    """

    def __init__(self, llm, retriever, web_search=None):
        """
        Initialize Corrective RAG.

        Args:
            llm: LLM instance
            retriever: Retriever instance
            web_search: Optional web search function
        """
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search

    def query(self, question: str, k: int = 5) -> RAGResponse:
        """
        Query with correction.

        Args:
            question: Question to ask
            k: Number of documents to retrieve

        Returns:
            RAGResponse with corrected answer
        """
        # Step 1: Retrieve documents
        docs = self.retriever.search(question, k=k)

        # Step 2: Evaluate retrieval quality
        quality, relevant_docs, irrelevant_docs = self._evaluate_retrieval(question, docs)

        # Step 3: Handle based on quality
        if quality == "correct":
            # Use relevant documents
            knowledge = self._refine_knowledge(question, relevant_docs)
        elif quality == "incorrect":
            # Rewrite query and search
            rewritten_query = self._rewrite_query(question)

            if self.web_search:
                web_results = self.web_search(rewritten_query)
                knowledge = web_results
            else:
                # Re-retrieve with rewritten query
                new_docs = self.retriever.search(rewritten_query, k=k)
                knowledge = self._refine_knowledge(question, new_docs)
        else:
            # Ambiguous - use what we have
            knowledge = self._refine_knowledge(question, relevant_docs)

        # Step 4: Generate answer
        answer = self._generate_answer(question, knowledge)

        return RAGResponse(
            answer=answer,
            sources=relevant_docs,
            confidence=0.8 if quality == "correct" else 0.6,
            metadata={
                "quality": quality,
                "rewritten": quality == "incorrect",
            }
        )

    def _evaluate_retrieval(
        self, question: str, docs: List[Document]
    ) -> Tuple[str, List[Document], List[Document]]:
        """Evaluate retrieval quality."""
        relevant = []
        irrelevant = []

        for doc in docs:
            prompt = f"""Evaluate if this document is relevant to the question.

Question: {question}

Document:
{doc.page_content[:500]}

Is this document relevant? Reply with only 'yes' or 'no'."""

            response = self.llm.generate(prompt).strip().lower()

            if response == "yes":
                relevant.append(doc)
            else:
                irrelevant.append(doc)

        # Determine quality
        if len(relevant) >= len(docs) * 0.5:
            quality = "correct"
        elif len(relevant) == 0:
            quality = "incorrect"
        else:
            quality = "ambiguous"

        return quality, relevant, irrelevant

    def _rewrite_query(self, question: str) -> str:
        """Rewrite query for better retrieval."""
        prompt = f"""Rewrite this search query to be more specific and effective.

Original query: {question}

Return ONLY the rewritten query."""

        return self.llm.generate(prompt).strip()

    def _refine_knowledge(self, question: str, docs: List[Document]) -> str:
        """Refine knowledge from documents."""
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Extract and refine the key information from these documents that is relevant to the question.

Question: {question}

Documents:
{context}

Refined knowledge:"""

        return self.llm.generate(prompt)

    def _generate_answer(self, question: str, knowledge: str) -> str:
        """Generate answer from refined knowledge."""
        prompt = f"""Answer the question based on the provided knowledge.

Knowledge:
{knowledge}

Question: {question}

Answer:"""

        return self.llm.generate(prompt)


class HyDE:
    """
    Hypothetical Document Embedding (HyDE).

    Generates hypothetical documents that would answer the query,
    then uses their embeddings for retrieval.

    Process:
    1. Generate hypothetical document that answers the query
    2. Embed the hypothetical document
    3. Retrieve similar real documents
    4. Generate answer from real documents

    Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    (Gao et al., 2022)

    Example:
        hyde = HyDE(llm=llm, embeddings=embeddings)
        results = hyde.search("What is Python?", documents)
    """

    def __init__(self, llm, embeddings):
        """
        Initialize HyDE.

        Args:
            llm: LLM instance
            embeddings: Embeddings instance
        """
        self.llm = llm
        self.embeddings = embeddings

    def generate_hypothetical(self, query: str) -> str:
        """
        Generate hypothetical document.

        Args:
            query: Search query

        Returns:
            Hypothetical document text
        """
        prompt = f"""Write a short, factual passage that would answer this question.
Write as if excerpting from a textbook or documentation.

Question: {query}

Passage:"""

        return self.llm.generate(prompt).strip()

    def search(
        self,
        query: str,
        documents: List[Document],
        k: int = 5
    ) -> List[Document]:
        """
        Search using HyDE.

        Args:
            query: Search query
            documents: Documents to search
            k: Number of results

        Returns:
            List of relevant documents
        """
        # Generate hypothetical document
        hypothetical = self.generate_hypothetical(query)

        # Get embeddings
        query_embedding = self.embeddings.embed_query(hypothetical)

        # Get document embeddings
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = _cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [documents[i] for i, _ in similarities[:k]]




class HyPE:
    """
    Hypothetical Prompt Embeddings (HyPE).

    Enhancement to HyPE that precomputes hypothetical prompts at indexing time.

    Unlike HyDE, HyPE does not require LLM calls at query time.

    Process:
    1. At indexing: Generate multiple hypothetical queries per chunk
    2. Embed hypothetical queries instead of document chunks
    3. At query: Match user query against stored hypothetical questions

    Reference: "HyPE: Hypothetical Prompt Embeddings" (2024)

    Example:
        hype = HyPE(llm=llm, embeddings=embeddings)
        hype.index(documents)
        results = hype.search("What is Python?")
    """

    def __init__(self, llm, embeddings, num_prompts: int = 3):
        """
        Initialize HyPE.

        Args:
            llm: LLM instance
            embeddings: Embeddings instance
            num_prompts: Number of hypothetical prompts per chunk
        """
        self.llm = llm
        self.embeddings = embeddings
        self.num_prompts = num_prompts

        # Storage
        self._chunks: List[Document] = []
        self._prompt_embeddings: List[List[float]] = []
        self._prompt_to_chunk: List[int] = []  # Maps prompt index to chunk index

    def index(self, documents: List[Document]) -> None:
        """
        Index documents with hypothetical prompts.

        Args:
            documents: Documents to index
        """
        self._chunks = documents
        self._prompt_embeddings = []
        self._prompt_to_chunk = []

        for chunk_idx, doc in enumerate(documents):
            # Generate hypothetical prompts
            prompts = self._generate_prompts(doc.page_content)

            # Embed prompts
            for prompt in prompts:
                embedding = self.embeddings.embed_query(prompt)
                self._prompt_embeddings.append(embedding)
                self._prompt_to_chunk.append(chunk_idx)

    def _generate_prompts(self, text: str) -> List[str]:
        """Generate hypothetical prompts for a chunk."""
        prompt = f"""Generate {self.num_prompts} different questions that this text could answer.

Text:
{text[:1000]}

Return one question per line, nothing else."""

        response = self.llm.generate(prompt)
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()]

        return questions[:self.num_prompts]

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search using HyPE.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant documents
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Calculate similarities with all prompt embeddings
        similarities = []
        for i, prompt_emb in enumerate(self._prompt_embeddings):
            sim = _cosine_similarity(query_embedding, prompt_emb)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get unique chunks
        seen_chunks = set()
        results = []

        for prompt_idx, _ in similarities:
            chunk_idx = self._prompt_to_chunk[prompt_idx]
            if chunk_idx not in seen_chunks:
                seen_chunks.add(chunk_idx)
                results.append(self._chunks[chunk_idx])

            if len(results) >= k:
                break

        return results




class ContextualCompression:
    """
    Contextual Compression.

    Compresses retrieved information while preserving query-relevant content.

    Uses an LLM to compress or summarize retrieved chunks,
    preserving key information relevant to the query.

    Example:
        compressor = ContextualCompression(llm=llm)
        compressed = compressor.compress(query, documents)
    """

    def __init__(self, llm):
        """
        Initialize Contextual Compression.

        Args:
            llm: LLM instance
        """
        self.llm = llm

    def compress(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Compress documents to relevant content.

        Args:
            query: Search query
            documents: Documents to compress

        Returns:
            List of compressed documents
        """
        compressed = []

        for doc in documents:
            # Extract relevant content
            relevant_content = self._extract_relevant(query, doc.page_content)

            if relevant_content.strip():
                compressed.append(Document(
                    page_content=relevant_content,
                    metadata=doc.metadata.copy()
                ))

        return compressed if compressed else documents[:1]

    def _extract_relevant(self, query: str, content: str) -> str:
        """Extract relevant content from document."""
        prompt = f"""Extract only the parts of this text that are relevant to the question.

Question: {query}

Text:
{content[:1500]}

Relevant content:"""

        return self.llm.generate(prompt).strip()


class DocumentAugmentation:
    """
    Document Augmentation.

    Generates additional questions for each document to improve retrieval.

    Uses an LLM to augment text dataset with all possible questions
    that can be asked to each document.

    Example:
        augmenter = DocumentAugmentation(llm=llm)
        augmented = augmenter.augment(documents)
    """

    def __init__(self, llm, num_questions: int = 5):
        """
        Initialize Document Augmentation.

        Args:
            llm: LLM instance
            num_questions: Number of questions per document
        """
        self.llm = llm
        self.num_questions = num_questions

    def augment(self, documents: List[Document]) -> List[Document]:
        """
        Augment documents with hypothetical questions.

        Args:
            documents: Documents to augment

        Returns:
            List of augmented documents
        """
        augmented = []

        for doc in documents:
            # Generate questions
            questions = self._generate_questions(doc.page_content)

            # Create augmented document
            augmented_content = doc.page_content + "\n\nRelated questions:\n"
            for q in questions:
                augmented_content += f"- {q}\n"

            augmented.append(Document(
                page_content=augmented_content,
                metadata=doc.metadata.copy()
            ))

        return augmented

    def _generate_questions(self, content: str) -> List[str]:
        """Generate questions for content."""
        prompt = f"""Generate {self.num_questions} different questions that this text could answer.

Text:
{content[:1000]}

Return one question per line, nothing else."""

        response = self.llm.generate(prompt)
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()]

        return questions[:self.num_questions]

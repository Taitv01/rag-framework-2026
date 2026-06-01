"""
Retrieval Agent
===============

Intelligent agent that decides whether and how to retrieve documents.

Features:
- Decides if retrieval is needed
- Selects retrieval strategy
- Handles multi-step retrieval

Usage:
    agent = RetrievalAgent(llm, retriever)
    result = agent.run("What is Python?")
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@dataclass
class RetrievalDecision:
    """Decision about whether to retrieve."""
    should_retrieve: bool
    reason: str
    query: Optional[str] = None


class RetrievalAgent:
    """
    Agent that makes intelligent retrieval decisions.

    Analyzes queries to determine:
    - Whether retrieval is needed
    - What type of retrieval to use
    - How to optimize the query

    Example:
        agent = RetrievalAgent(llm=llm, retriever=retriever)

        # Single query
        result = agent.run("What is Python?")

        # With conversation context
        result = agent.run(
            "Tell me more",
            conversation_history=[...]
        )
    """

    def __init__(
        self,
        llm,
        retriever,
        max_retrieval_attempts: int = 3,
    ):
        """
        Initialize retrieval agent.

        Args:
            llm: LLM instance
            retriever: Retriever instance
            max_retrieval_attempts: Maximum retrieval attempts
        """
        self.llm = llm
        self.retriever = retriever
        self.max_retrieval_attempts = max_retrieval_attempts

    def run(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Run the retrieval agent.

        Args:
            question: Question to answer
            conversation_history: Optional conversation history
            k: Number of documents to retrieve

        Returns:
            Dict with answer and metadata
        """
        # Step 1: Decide if retrieval is needed
        decision = self._make_retrieval_decision(question, conversation_history)

        if not decision.should_retrieve:
            # Answer without retrieval
            answer = self._generate_direct_answer(question, conversation_history)
            return {
                "answer": answer,
                "retrieved": False,
                "reason": decision.reason,
            }

        # Step 2: Retrieve documents
        query = decision.query or question
        docs = self._retrieve_with_retry(query, k)

        # Step 3: Generate answer
        answer = self._generate_answer(question, docs, conversation_history)

        return {
            "answer": answer,
            "retrieved": True,
            "reason": decision.reason,
            "query_used": query,
            "documents": [
                {"content": doc.page_content[:200], "metadata": doc.metadata}
                for doc in docs
            ],
        }

    def _make_retrieval_decision(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RetrievalDecision:
        """Decide whether retrieval is needed."""
        # Build context
        context = ""
        if conversation_history:
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-3:]  # Last 3 messages
            ])

        prompt = f"""You are a retrieval decision agent. Decide if the question requires retrieving documents from a knowledge base.

Consider:
1. Is this a factual question that needs specific information?
2. Is this a conversational question that can be answered from context?
3. Is this a general knowledge question?

{f"Conversation context:{chr(10)}{context}" if context else ""}

Question: {question}

Should we retrieve documents? Answer 'yes' or 'no' and explain why.
Format: yes/no: reason"""

        response = self.llm.generate(prompt)
        response = response.strip().lower()

        # Parse response
        if response.startswith("yes"):
            reason = response.split(":", 1)[-1].strip() if ":" in response else "Retrieval needed"
            return RetrievalDecision(
                should_retrieve=True,
                reason=reason,
                query=question
            )
        else:
            reason = response.split(":", 1)[-1].strip() if ":" in response else "Direct answer possible"
            return RetrievalDecision(
                should_retrieve=False,
                reason=reason
            )

    def _retrieve_with_retry(
        self,
        query: str,
        k: int
    ) -> List[Document]:
        """Retrieve with query optimization retry."""
        docs = self.retriever.search(query, k=k)

        # If no results, try query transformation
        if not docs:
            for _ in range(self.max_retrieval_attempts - 1):
                transformed_query = self._transform_query(query)
                docs = self.retriever.search(transformed_query, k=k)
                if docs:
                    break

        return docs

    def _transform_query(self, query: str) -> str:
        """Transform query for better retrieval."""
        prompt = f"""Transform this search query to retrieve better results.

Original query: {query}

Return ONLY the transformed query, nothing else."""

        return self.llm.generate(prompt).strip()

    def _generate_answer(
        self,
        question: str,
        docs: List[Document],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate answer from documents."""
        # Build context
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        # Build conversation context
        conv_context = ""
        if conversation_history:
            conv_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-3:]
            ])

        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the question.

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say so
3. Be concise and accurate

{f"Previous conversation:{chr(10)}{conv_context}" if conv_context else ""}

Context:
{context}

Question: {question}"""

        return self.llm.generate(prompt)

    def _generate_direct_answer(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate answer without retrieval."""
        # Build conversation context
        conv_context = ""
        if conversation_history:
            conv_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-3:]
            ])

        prompt = f"""You are a helpful AI assistant. Answer the question based on your knowledge.

{f"Previous conversation:{chr(10)}{conv_context}" if conv_context else ""}

Question: {question}"""

        return self.llm.generate(prompt)

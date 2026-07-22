"""
Agentic RAG
===========

Agent-based RAG using LangGraph for intelligent retrieval decisions.

Features:
- LLM decides whether to retrieve
- Document relevance grading (bilingual)
- Query rewriting loop (Vietnamese-aware)
- Multi-step reasoning

Architecture:
START → generate_query_or_respond → [tool_calls?] → retrieve → grade_documents
            ↑                                        ↓              ↓
            ←── rewrite_question ←── [irrelevant]    [relevant]
                                                                ↓
                                                          generate_answer → END

Usage:
    rag = AgenticRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("Thạch Sanh là ai?")
"""

import logging
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Document relevance grading schema."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRAG:
    """
    Agent-based RAG with LangGraph.

    Uses an intelligent agent to:
    - Decide whether retrieval is needed
    - Grade document relevance
    - Rewrite queries for better retrieval
    - Generate answers from relevant context

    Example:
        rag = AgenticRAG(
            llm_provider="openai",
            llm_model="gpt-4o"
        )

        rag.add_documents(["documents/"])

        # Query (agent decides whether to retrieve)
        answer = rag.query("What is Python?")

        # Query with conversation history
        answer = rag.query(
            "Tell me more about that",
            conversation_history=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language..."}
            ]
        )
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: str = "keepitreal/vietnamese-sbert",
        vector_store_provider: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        max_retries: int = 3,
    ):
        """
        Initialize Agentic RAG.

        Args:
            llm_provider: LLM provider
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider
            embedding_model: Embedding model name
            vector_store_provider: Vector store provider
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            retrieval_k: Number of documents to retrieve
            max_retries: Maximum query rewrite retries
        """
        from src.core.document_loader import DocumentLoader
        from src.core.text_splitter import TextSplitter
        from src.core.embeddings import EmbeddingsManager
        from src.core.vector_store import VectorStoreManager
        from src.core.llm import LLMManager

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = EmbeddingsManager(
            provider=embedding_provider,
            model_name=embedding_model,
        )
        self.vector_store = VectorStoreManager(
            provider=vector_store_provider,
            embeddings=self.embeddings,
        )
        self.llm = LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        self.retrieval_k = retrieval_k
        self.max_retries = max_retries

        # Track documents
        self._documents = []
        self._chunks = []

        # Initialize graph
        self._graph = None
        self._retriever_tool = None

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the knowledge base.

        Args:
            sources: File path(s) or directory path(s)
            metadata: Additional metadata

        Returns:
            Number of chunks added
        """
        # Normalize to list
        if isinstance(sources, (str, Path)):
            sources = [sources]

        # Load documents
        all_docs = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                docs = self.document_loader.load_directory(source, metadata=metadata)
            else:
                docs = self.document_loader.load(source, metadata=metadata)
            all_docs.extend(docs)

        self._documents.extend(all_docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(all_docs)
        self._chunks.extend(chunks)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        # Create retriever tool
        self._create_retriever_tool()

        # Build graph
        self._build_graph()

        return len(chunks)

    def _create_retriever_tool(self):
        """Create retriever tool for the agent."""
        retriever = self.vector_store.get_retriever(k=self.retrieval_k)

        @tool
        def retrieve_documents(query: str) -> str:
            """Search and return relevant documents from the knowledge base."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        self._retriever_tool = retrieve_documents

    def _build_graph(self):
        """Build LangGraph workflow."""
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.graph import MessagesState
            from langgraph.prebuilt import ToolNode
        except ImportError:
            raise ImportError(
                "langgraph is required for AgenticRAG. "
                "Install it with: pip install langgraph"
            )

        # Define workflow
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self._retriever_tool]))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")

        # Conditional edge: decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._route_on_tool_calls,
            {"tools": "retrieve", END: END},
        )

        # Conditional edge: grade documents
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
            },
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile graph
        self._graph = workflow.compile()

    def _generate_query_or_respond(self, state: Dict) -> Dict:
        """Generate response or decide to retrieve."""
        from langgraph.graph import MessagesState

        response = self.llm.bind_tools([self._retriever_tool]).invoke(
            state["messages"]
        )

        return {"messages": [response]}

    def _route_on_tool_calls(self, state: Dict) -> Literal["tools", "__end__"]:
        """Route based on whether tool calls were made."""
        from langgraph.graph import END

        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return END

    def _get_question_from_state(self, state: Dict) -> str:
        """Extract the original question from state messages.

        Handles conversation history by finding the last HumanMessage
        before the tool calls.
        """
        messages = state["messages"]
        # Find the last human message (the question)
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        # Fallback to first message
        return messages[0].content

    def _grade_documents(self, state: Dict) -> Literal["generate_answer", "rewrite_question"]:
        """Grade document relevance."""
        question = self._get_question_from_state(state)
        context = state["messages"][-1].content

        prompt = f"""You are a document relevance grader / Bạn là người đánh giá tài liệu.
Determine if the retrieved documents are relevant to the question.
Xác định tài liệu có liên quan đến câu hỏi không.

Question / Câu hỏi: {question}

Retrieved documents / Tài liệu:
{context}

Are these documents relevant? Answer only 'yes' or 'no'.
Tài liệu có liên quan không? Chỉ trả lời 'yes' hoặc 'no'."""

        response = self.llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )

        if response.binary_score in ("yes", "có"):
            return "generate_answer"
        else:
            return "rewrite_question"

    def _rewrite_question(self, state: Dict) -> Dict:
        """Rewrite question for better retrieval."""
        question = self._get_question_from_state(state)

        prompt = f"""You are a search query optimizer / Bạn là người tối ưu hóa truy vấn.
Transform this question into a better search query.
Chuyển đổi câu hỏi thành truy vấn tốt hơn.

Original question / Câu hỏi gốc: {question}

Return ONLY the optimized search query, nothing else."""

        response = self.llm.generate(prompt)

        return {"messages": [HumanMessage(content=response)]}

    def _generate_answer(self, state: Dict) -> Dict:
        """Generate answer from context."""
        question = self._get_question_from_state(state)
        context = state["messages"][-1].content

        prompt = f"""You are a helpful AI assistant / Bạn là trợ lý AI hữu ích.
Use the provided context to answer the question.
Sử dụng ngữ cảnh để trả lời câu hỏi.

Rules / Quy tắc:
1. Answer based ONLY on the provided context / Chỉ trả lời dựa trên ngữ cảnh
2. If the context doesn't contain the answer, say so / Nếu không đủ thông tin, nói rõ
3. Be concise and accurate / Ngắn gọn và chính xác
4. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}"""

        response = self.llm.generate(prompt)

        return {"messages": [AIMessage(content=response)]}

    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Query using the agentic RAG pipeline.

        Args:
            question: Question to ask
            conversation_history: Optional conversation history

        Returns:
            Answer string
        """
        if self._graph is None:
            raise RuntimeError(
                "No documents loaded. Call add_documents() first."
            )

        # Build messages
        messages = []

        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

        # Add current question
        messages.append(HumanMessage(content=question))

        # Run graph
        result = self._graph.invoke({"messages": messages})

        # Extract answer
        answer = result["messages"][-1].content

        return answer

    def query_with_trace(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with execution trace.

        Args:
            question: Question to ask
            conversation_history: Optional conversation history

        Returns:
            Dict with answer and execution trace
        """
        if self._graph is None:
            raise RuntimeError(
                "No documents loaded. Call add_documents() first."
            )

        # Build messages
        messages = []

        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=question))

        # Run graph with tracing
        trace = []
        result = None

        for event in self._graph.stream({"messages": messages}):
            for node, output in event.items():
                trace.append({
                    "node": node,
                    "output": output,
                })
            result = event

        # Extract answer from final result
        if result:
            last_output = list(result.values())[-1]
            answer = last_output["messages"][-1].content
        else:
            answer = "No answer generated"

        return {
            "answer": answer,
            "trace": trace,
            "question": question,
        }

    def check_hallucination(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Grade whether an answer is factually grounded in the provided context (Hallucination Check).

        Args:
            context: Retrieved context text
            answer: Generated answer text

        Returns:
            Dict containing is_grounded boolean, hallucination_score (0.0 to 1.0), and reasoning
        """
        prompt = f"""Bạn là chuyên gia kiểm tra ảo giác dữ liệu (Hallucination Checker).
Nhiệm vụ: Đánh giá xem câu trả lời có được căn cứ HOÀN TOÀN vào ngữ cảnh cung cấp hay không (không tự bịa ra thông tin mới).

Ngữ cảnh (Context):
{context}

Câu trả lời (Answer):
{answer}

Quy tắc:
- Trả lời "Grounded: yes" nếu câu trả lời hoàn toàn chính xác dựa vào ngữ cảnh.
- Trả lời "Grounded: no" nếu câu trả lời chứa thông tin mâu thuẫn hoặc không có trong ngữ cảnh.

Đánh giá:"""
        try:
            response = self.llm.generate(prompt)
            is_grounded = "yes" in response.lower() or "có" in response.lower() or "đúng" in response.lower()
            score = 0.0 if is_grounded else 1.0
            return {
                "is_grounded": is_grounded,
                "hallucination_score": score,
                "reasoning": response.strip(),
            }
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return {
                "is_grounded": True,
                "hallucination_score": 0.0,
                "reasoning": f"Fallback check: {e}",
            }

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

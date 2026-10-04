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
            ←── rewrite_question ←── [irrelevant]    [relevant, or max_retries reached]
                                                                ↓
                                                          generate_answer → END

The loop is bounded twice: at most ``max_retries`` rewrites, and LangGraph's
``recursion_limit`` as a safety net. Grading and the answer always use the
user's question, not a rewritten search query.

Usage:
    rag = AgenticRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("Thạch Sanh là ai?")

    # Or search an index another pipeline owns (hybrid, reranking, parents):
    advanced = AdvancedRAG()
    agent = AgenticRAG(search=advanced.retrieve, llm=advanced.llm)
"""

import json
import logging
import re
from typing import Callable, List, Optional, Dict, Any, Union, Literal
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Answer when the agent loop runs past its recursion limit.
GAVE_UP_ANSWER = (
    "Tôi không có đủ thông tin trong tài liệu để trả lời câu hỏi này. / "
    "I don't have enough information in the documents to answer this question."
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
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: Optional[str] = None,
        vector_store_provider: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        max_retries: int = 3,
        recursion_limit: Optional[int] = None,
        search: Optional[Callable[[str, int], List[Document]]] = None,
        llm=None,
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
            max_retries: Maximum query rewrites before answering from what was found
            recursion_limit: LangGraph step limit (default: enough for max_retries)
            search: Shared ``search(query, k) -> documents`` over an index another
                pipeline owns (e.g. ``AdvancedRAG.retrieve``); no embedding model
                or vector store is loaded and add_documents() is not used
            llm: Shared LLMManager instead of creating one
        """
        from src.core.document_loader import DocumentLoader
        from src.core.text_splitter import TextSplitter
        from src.core.llm import LLMManager

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._search_fn = search
        self.embeddings = None
        self.vector_store = None
        if search is None:
            from src.core.embeddings import EmbeddingsManager
            from src.core.vector_store import VectorStoreManager

            self.embeddings = EmbeddingsManager(
                provider=embedding_provider,
                model_name=embedding_model,
            )
            self.vector_store = VectorStoreManager(
                provider=vector_store_provider,
                embeddings=self.embeddings,
            )
        self.llm = llm or LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        self.retrieval_k = retrieval_k
        self.max_retries = max(0, max_retries)
        # Each rewrite costs three graph steps (decide, retrieve, rewrite); the
        # final round three more. The default leaves two steps of headroom.
        self.recursion_limit = recursion_limit or 3 * (self.max_retries + 1) + 2

        # Track documents
        self._documents = []
        self._chunks = []

        # Initialize graph
        self._graph = None
        self._retriever_tool = None
        if search is not None:
            # The shared index is searchable already.
            self._create_retriever_tool()
            self._build_graph()

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
        if self._search_fn is not None:
            raise RuntimeError(
                "This AgenticRAG searches a shared index: add documents to the "
                "pipeline that owns it."
            )

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

    def _search(self, query: str) -> List[Document]:
        if self._search_fn is not None:
            return list(self._search_fn(query, self.retrieval_k))
        return self.vector_store.similarity_search(query, k=self.retrieval_k)

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        """Passages labelled [S1], [S2]... with their source, as AdvancedRAG does."""
        parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = metadata.get("source") or metadata.get("file_name") or metadata.get("url") or f"Document {i}"
            parts.append(f"[S{i}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(parts)

    def _create_retriever_tool(self):
        """Create retriever tool for the agent."""

        # The model reads the labelled text; the documents travel as the
        # message artifact so callers keep their metadata for citations.
        @tool(response_format="content_and_artifact")
        def retrieve_documents(query: str):
            """Search and return relevant documents from the knowledge base."""
            docs = self._search(query)
            return self._format_context(docs), docs

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

        class AgentState(MessagesState):
            question: str  # the user's question; rewrites never replace it
            rewrites: int

        # Define workflow
        workflow = StateGraph(AgentState)

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
        """The user's question (rewritten search queries are not questions)."""
        if state.get("question"):
            return state["question"]
        messages = state["messages"]
        # States built without a question: the last human message.
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return messages[0].content

    def _grade_documents(self, state: Dict) -> Literal["generate_answer", "rewrite_question"]:
        """Grade document relevance; after max_retries rewrites, answer anyway."""
        question = self._get_question_from_state(state)
        context = state["messages"][-1].content

        relevant = False
        if context.strip():
            prompt = f"""You are a document relevance grader / Bạn là người đánh giá tài liệu.
Determine if the retrieved documents are relevant to the question.
Xác định tài liệu có liên quan đến câu hỏi không.

Question / Câu hỏi: {question}

Retrieved documents / Tài liệu:
{context}

Are these documents relevant? Answer only 'yes' or 'no'.
Tài liệu có liên quan không? Chỉ trả lời 'yes' hoặc 'no'."""
            # Plain text, not structured output: works with every provider.
            reply = str(self.llm.generate(prompt)).strip().casefold()
            relevant = reply.startswith(("yes", "có"))

        if relevant:
            return "generate_answer"
        if state.get("rewrites", 0) >= self.max_retries:
            # The answer prompt tells the model to say when the context lacks the answer.
            logger.info(f"No relevant documents after {self.max_retries} rewrites; answering anyway")
            return "generate_answer"
        return "rewrite_question"

    def _rewrite_question(self, state: Dict) -> Dict:
        """Rewrite the user's question into a search query not tried yet."""
        question = self._get_question_from_state(state)
        tried = [
            call["args"].get("query", "")
            for msg in state["messages"] if isinstance(msg, AIMessage)
            for call in (msg.tool_calls or [])
        ]
        tried_block = "\n".join(f"- {query}" for query in tried if query)

        prompt = f"""You are a search query optimizer / Bạn là người tối ưu hóa truy vấn.
Transform this question into a better search query.
Chuyển đổi câu hỏi thành truy vấn tốt hơn.

Original question / Câu hỏi gốc: {question}
"""
        if tried_block:
            prompt += f"""
These queries found nothing relevant; write a different one / Các truy vấn sau không tìm được gì, hãy viết truy vấn khác:
{tried_block}
"""
        prompt += "\nReturn ONLY the optimized search query, nothing else."

        response = self.llm.generate(prompt)

        return {
            "messages": [HumanMessage(content=response)],
            "rewrites": state.get("rewrites", 0) + 1,
        }

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
5. Cite the sources you use as [S1], [S2] / Trích nguồn đã dùng dạng [S1], [S2]

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}"""

        response = self.llm.generate(prompt)

        return {"messages": [AIMessage(content=response)]}

    def _initial_state(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if self._graph is None:
            raise RuntimeError(
                "No documents loaded. Call add_documents() first."
            )

        messages = []
        for msg in conversation_history or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=question))

        return {"messages": messages, "question": question, "rewrites": 0}

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
        return self.query_with_trace(question, conversation_history)["answer"]

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
            Dict with the answer, the execution trace, ``sources`` (the
            passages of the last retrieval, labelled as in the answer's [S#]
            citations) and ``rewrites``
        """
        from langgraph.errors import GraphRecursionError

        state = self._initial_state(question, conversation_history)
        trace = []
        answer = None
        docs: List[Document] = []
        rewrites = 0

        try:
            for event in self._graph.stream(state, config={"recursion_limit": self.recursion_limit}):
                for node, output in event.items():
                    trace.append({
                        "node": node,
                        "output": output,
                    })
                    output = output or {}
                    rewrites = output.get("rewrites", rewrites)
                    messages = output.get("messages") or []
                    for message in messages:
                        if isinstance(message, ToolMessage):
                            docs = list(message.artifact or [])
                    if messages:
                        answer = messages[-1].content
        except GraphRecursionError:
            logger.warning(f"Agent loop hit recursion_limit={self.recursion_limit}; giving up")
            answer = GAVE_UP_ANSWER

        return {
            "answer": answer if answer is not None else GAVE_UP_ANSWER,
            "trace": trace,
            "question": question,
            "sources": self._format_sources(docs),
            "rewrites": rewrites,
        }

    @staticmethod
    def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, 1):
            metadata = {k: v for k, v in (doc.metadata or {}).items() if k != "parent_text"}
            content = doc.page_content
            sources.append({
                "source_id": f"S{i}",
                "source": metadata.get("source") or metadata.get("file_name") or metadata.get("url") or f"Document {i}",
                "content": content[:300] + "..." if len(content) > 300 else content,
                "metadata": metadata,
            })
        return sources

    def check_hallucination(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Grade whether an answer is factually grounded in the provided context (Hallucination Check).

        Args:
            context: Retrieved context text
            answer: Generated answer text

        Returns:
            Dict containing is_grounded boolean, hallucination_score (0.0 to 1.0), and reasoning
        """
        if not context.strip() and answer.strip():
            return {
                "is_grounded": False,
                "hallucination_score": 1.0,
                "reasoning": "Cannot ground a non-empty answer without context.",
            }

        prompt = f"""Bạn là chuyên gia kiểm tra ảo giác dữ liệu (Hallucination Checker).
Nhiệm vụ: Đánh giá xem câu trả lời có được căn cứ HOÀN TOÀN vào ngữ cảnh cung cấp hay không (không tự bịa ra thông tin mới).

Ngữ cảnh (Context):
{context}

Câu trả lời (Answer):
{answer}

Quy tắc:
- grounded=true nếu câu trả lời hoàn toàn chính xác dựa vào ngữ cảnh.
- grounded=false nếu câu trả lời chứa thông tin mâu thuẫn hoặc không có trong ngữ cảnh.
- hallucination_score nằm trong khoảng 0.0 (hoàn toàn có căn cứ) đến 1.0 (hoàn toàn không có căn cứ).
- Chỉ trả về JSON hợp lệ theo đúng schema sau, không thêm markdown:
  {{"grounded": true, "hallucination_score": 0.0, "reasoning": "lý do ngắn gọn"}}

Đánh giá:"""
        try:
            response = str(self.llm.generate(prompt)).strip()
            is_grounded, score, reasoning = self._parse_grounding_response(response)
            return {
                "is_grounded": is_grounded,
                "hallucination_score": score,
                "reasoning": reasoning,
            }
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return {
                "is_grounded": False,
                "hallucination_score": 1.0,
                "reasoning": f"Grounding check unavailable: {e}",
            }

    @staticmethod
    def _parse_grounding_response(response: str) -> tuple[bool, float, str]:
        """Parse a structured grounding grade with a strict legacy-text fallback."""
        if not response:
            raise ValueError("Empty grounding response")

        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if json_match:
            payload = json.loads(json_match.group(0))
            grounded = payload.get("grounded")
            if not isinstance(grounded, bool):
                raise ValueError("'grounded' must be a JSON boolean")

            raw_score = payload.get("hallucination_score", 0.0 if grounded else 1.0)
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("'hallucination_score' must be a number")

            score = max(0.0, min(1.0, float(raw_score)))
            reasoning = str(payload.get("reasoning") or response).strip()
            return grounded, score, reasoning

        # Backward compatibility for existing prompts/providers. Anchoring the
        # label avoids false positives such as "Grounded: no, not yes".
        label_match = re.search(
            r"(?im)^\s*grounded\s*:\s*(yes|no|có|không|đúng|sai)\b",
            response,
        )
        if not label_match:
            raise ValueError("Unrecognized grounding response format")

        grounded = label_match.group(1).casefold() in {"yes", "có", "đúng"}
        return grounded, 0.0 if grounded else 1.0, response

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

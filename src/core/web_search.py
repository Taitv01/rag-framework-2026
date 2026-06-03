"""
Web Search
==========

Web search fallback for RAG when local document retrieval is insufficient.

Safety-first design:
- Web results are clearly labeled with source_type="web"
- Relevance verification before use
- Configurable providers (Tavily, DuckDuckGo)
- Explicit disclosure in answers
- OFF by default — users must opt in

Usage:
    from src.core.web_search import SafeWebSearcher, DuckDuckGoSearchProvider

    provider = DuckDuckGoSearchProvider()
    searcher = SafeWebSearcher(provider, llm=llm)

    results = searcher.search("Thạch Sanh là ai?", num_results=3)
    docs = searcher.to_documents(results)
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    full_content: Optional[str] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSearchProvider:
    """
    Base class for web search providers.

    Subclasses implement the actual search API calls.
    """

    def search(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """
        Search the web for the given query.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of WebSearchResult objects
        """
        raise NotImplementedError


class DuckDuckGoSearchProvider(WebSearchProvider):
    """
    DuckDuckGo search provider.

    Free, no API key required. Uses the duckduckgo-search library.

    Example:
        provider = DuckDuckGoSearchProvider()
        results = provider.search("Thạch Sanh", num_results=5)
    """

    def __init__(self, region: str = "vn", safesearch: str = "moderate"):
        """
        Initialize DuckDuckGo search.

        Args:
            region: Search region (e.g., "vn", "us", "wt-wt")
            safesearch: Safe search level ("off", "moderate", "strict")
        """
        self.region = region
        self.safesearch = safesearch
        self._available = None

    def _check_availability(self) -> bool:
        """Check if duckduckgo-search is installed."""
        if self._available is None:
            try:
                from duckduckgo_search import DDGS
                self._available = True
            except ImportError:
                logger.warning(
                    "duckduckgo-search not installed. "
                    "Install with: pip install duckduckgo-search"
                )
                self._available = False
        return self._available

    def search(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using DuckDuckGo."""
        if not self._check_availability():
            return []

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=num_results,
                ):
                    results.append(WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", "")),
                    ))

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []


class TavilySearchProvider(WebSearchProvider):
    """
    Tavily search provider.

    Optimized for RAG — returns clean, relevant content.
    Requires API key (set TAVILY_API_KEY env var or pass directly).

    Example:
        provider = TavilySearchProvider(api_key="tvly-...")
        results = provider.search("Thạch Sanh", num_results=5)
    """

    def __init__(self, api_key: Optional[str] = None, search_depth: str = "basic"):
        """
        Initialize Tavily search.

        Args:
            api_key: Tavily API key (or from TAVILY_API_KEY env var)
            search_depth: "basic" (fast) or "advanced" (thorough)
        """
        self._api_key = api_key
        self.search_depth = search_depth
        self._available = None

    def _get_api_key(self) -> Optional[str]:
        """Get API key from param or environment."""
        if self._api_key:
            return self._api_key
        import os
        return os.getenv("TAVILY_API_KEY")

    def _check_availability(self) -> bool:
        """Check if tavily-python is installed and API key is set."""
        if self._available is None:
            try:
                from tavily import TavilyClient
                self._available = bool(self._get_api_key())
                if not self._available:
                    logger.warning("Tavily API key not set. Set TAVILY_API_KEY env var.")
            except ImportError:
                logger.warning(
                    "tavily-python not installed. "
                    "Install with: pip install tavily-python"
                )
                self._available = False
        return self._available

    def search(self, query: str, num_results: int = 5) -> List[WebSearchResult]:
        """Search using Tavily."""
        if not self._check_availability():
            return []

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self._get_api_key())
            response = client.search(
                query=query,
                max_results=num_results,
                search_depth=self.search_depth,
                include_answer=True,
            )

            results = []

            # Add the synthesized answer if available
            if response.get("answer"):
                results.append(WebSearchResult(
                    title="Tavily Synthesized Answer",
                    url="",
                    snippet=response["answer"],
                    metadata={"is_synthesized": True},
                ))

            # Add individual results
            for r in response.get("results", []):
                results.append(WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", r.get("snippet", "")),
                    full_content=r.get("raw_content"),
                    relevance_score=r.get("score", 0.0),
                ))

            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []


class SafeWebSearcher:
    """
    Web search with safety verification.

    Wraps a WebSearchProvider with:
    - Relevance verification (LLM checks if results are actually relevant)
    - Source labeling (all results marked as "web" source)
    - Content truncation (avoids injecting too much noise)
    - Disclosure (results are clearly identified as web-sourced)

    Example:
        provider = DuckDuckGoSearchProvider()
        searcher = SafeWebSearcher(provider, llm=llm)

        results = searcher.search("Thạch Sanh là ai?")
        docs = searcher.to_documents(results)
    """

    def __init__(
        self,
        provider: WebSearchProvider,
        llm=None,
        max_content_length: int = 1000,
        verify_relevance: bool = True,
    ):
        """
        Initialize safe web searcher.

        Args:
            provider: WebSearchProvider instance
            llm: Optional LLMManager for relevance verification
            max_content_length: Max chars per result content
            verify_relevance: Whether to verify result relevance with LLM
        """
        self.provider = provider
        self.llm = llm
        self.max_content_length = max_content_length
        self.verify_relevance = verify_relevance

    def search(
        self,
        query: str,
        num_results: int = 5,
        verify: Optional[bool] = None,
    ) -> List[WebSearchResult]:
        """
        Search and optionally verify results.

        Args:
            query: Search query
            num_results: Number of results to fetch
            verify: Override verify_relevance setting

        Returns:
            List of relevant WebSearchResult objects
        """
        should_verify = verify if verify is not None else self.verify_relevance

        # Search
        results = self.provider.search(query, num_results)

        if not results:
            logger.info(f"No web results found for: {query}")
            return []

        # Filter out synthesized answers from verification
        synthesized = [r for r in results if r.metadata.get("is_synthesized")]
        real_results = [r for r in results if not r.metadata.get("is_synthesized")]

        # Verify relevance
        if should_verify and self.llm and real_results:
            real_results = self._verify_relevance(query, real_results)

        # Combine: synthesized answers first, then verified results
        return synthesized + real_results

    def _verify_relevance(
        self, query: str, results: List[WebSearchResult]
    ) -> List[WebSearchResult]:
        """Verify each result is relevant to the query."""
        verified = []

        for result in results:
            try:
                prompt = f"""Is this web search result relevant to the question?
Kết quả tìm kiếm này có liên quan đến câu hỏi không?

Question / Câu hỏi: {query}

Result / Kết quả:
Title: {result.title}
Content: {result.snippet[:500]}

Reply with only 'yes' or 'no'.
Chỉ trả lời 'yes' hoặc 'no'."""

                response = self.llm.generate(prompt).strip().lower()

                if response in ("yes", "có", "đúng"):
                    result.relevance_score = 1.0
                    verified.append(result)
                else:
                    logger.debug(f"Filtered irrelevant result: {result.title[:50]}")

            except Exception as e:
                logger.warning(f"Relevance verification failed: {e}")
                # Include on failure (fail-open for web results)
                verified.append(result)

        return verified

    def search_and_verify(
        self, query: str, num_results: int = 3
    ) -> Tuple[List[WebSearchResult], float]:
        """
        Search, verify, and return with confidence score.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            Tuple of (results, confidence_score)
        """
        results = self.search(query, num_results)

        if not results:
            return [], 0.0

        # Calculate confidence based on number of verified results
        non_synthesized = [r for r in results if not r.metadata.get("is_synthesized")]
        verified_count = sum(1 for r in non_synthesized if r.relevance_score > 0)
        total_count = len(non_synthesized)

        confidence = verified_count / total_count if total_count > 0 else 0.0

        return results, confidence

    def to_documents(self, results: List[WebSearchResult]) -> List[Document]:
        """
        Convert web results to LangChain Document objects.

        Each document is labeled with source_type="web" and the URL.

        Args:
            results: List of WebSearchResult objects

        Returns:
            List of Document objects with web metadata
        """
        docs = []

        for result in results:
            content = result.snippet or ""
            if result.full_content:
                content = result.full_content[:self.max_content_length]
            elif len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."

            metadata = {
                "source": result.url or "web search",
                "source_type": "web",
                "url": result.url,
                "title": result.title,
                "web_search_result": True,
                "relevance_score": result.relevance_score,
            }

            # Prefix with source disclosure
            disclosed_content = (
                f"[Web Search Result: {result.title}]\n"
                f"[URL: {result.url}]\n\n"
                f"{content}"
            )

            if result.metadata.get("is_synthesized"):
                metadata["is_synthesized"] = True
                disclosed_content = (
                    f"[Web Search - Synthesized Answer]\n\n"
                    f"{content}"
                )

            docs.append(Document(
                page_content=disclosed_content,
                metadata=metadata,
            ))

        return docs

    def create_web_answer_prompt(
        self,
        question: str,
        web_docs: List[Document],
        local_docs: Optional[List[Document]] = None,
    ) -> str:
        """
        Create a prompt that clearly separates web and local sources.

        Args:
            question: User's question
            web_docs: Web-sourced documents
            local_docs: Local document-sourced documents

        Returns:
            Formatted prompt with clear source labeling
        """
        context_parts = []

        if local_docs:
            context_parts.append("=== LOCAL DOCUMENTS / TÀI LIỆU CỤC BỘ ===")
            for i, doc in enumerate(local_docs, 1):
                context_parts.append(f"[Local Document {i}]\n{doc.page_content}")
            context_parts.append("")

        if web_docs:
            context_parts.append("=== WEB SEARCH RESULTS / KẾT QUẢ TÌM KIẾM WEB ===")
            context_parts.append(
                "NOTE: The following information comes from web search.\n"
                "LƯU Ý: Thông tin sau đến từ tìm kiếm web.\n"
                "Please verify accuracy and cite the source URL.\n"
                "Vui lòng kiểm tra tính chính xác và trích dẫn URL nguồn.\n"
            )
            for i, doc in enumerate(web_docs, 1):
                context_parts.append(f"[Web Result {i}]\n{doc.page_content}")
            context_parts.append("")

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful assistant / Bạn là trợ lý hữu ích.
Answer the question using the provided context.
Trả lời câu hỏi bằng ngữ cảnh được cung cấp.

Rules / Quy tắc:
1. Answer based on the provided context / Trả lời dựa trên ngữ cảnh
2. When using web search results, ALWAYS mention the source URL / Khi dùng kết quả web, LUÔN trích dẫn URL
3. If information conflicts between local and web sources, prefer local sources / Nếu thông tin xung đột, ưu tiên tài liệu cục bộ
4. If uncertain, say so / Nếu không chắc chắn, hãy nói rõ
5. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}

Answer / Câu trả lời:"""

        return prompt


def create_web_searcher(
    provider: str = "duckduckgo",
    llm=None,
    api_key: Optional[str] = None,
    **kwargs,
) -> SafeWebSearcher:
    """
    Factory function to create a SafeWebSearcher.

    Args:
        provider: Provider name ("duckduckgo" or "tavily")
        llm: Optional LLMManager for relevance verification
        api_key: API key (for Tavily)
        **kwargs: Additional provider arguments

    Returns:
        SafeWebSearcher instance
    """
    if provider == "tavily":
        search_provider = TavilySearchProvider(api_key=api_key, **kwargs)
    elif provider == "duckduckgo":
        search_provider = DuckDuckGoSearchProvider(**kwargs)
    else:
        raise ValueError(f"Unknown web search provider: {provider}")

    return SafeWebSearcher(provider=search_provider, llm=llm)

"""
Knowledge graph index (GraphRAG)
================================

Entities and relationships extracted by the LLM, merged across documents and
grouped into communities, each with an LLM-written report: Microsoft
GraphRAG's index, sized for this pipeline.

- Extraction costs one LLM call per extraction unit (a document, or a slice
  of up to ``unit_chars`` characters of a long one), not one per chunk, and
  is redone only when a document's text changes.
- Communities are found with Louvain (networkx) on the merged entity graph.
  Each community of at least ``min_community_size`` entities gets one report,
  rewritten only when its entities, descriptions or relationships change.
- Queries call no LLM: entity profiles and community reports are searched,
  then reranked with the passages (AdvancedRAG keeps those that would rank in
  the top k).

Storage, beside the chunks and persisted like them:
``<collection>__graph`` holds one profile per entity and document (name,
type, description and that document's relationships) with the document's
metadata, so metadata filters and deletions apply as they do to passages;
``<collection>__communities`` holds the reports. At start-up the graph is
rebuilt from the profiles without LLM calls.
"""

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from langchain_core.documents import Document

from src.core.vector_store import metadata_matches
from src.rag.document_cards import (
    _NOT_DOCUMENT_LEVEL,
    MAX_CONSECUTIVE_CARD_FAILURES,
    content_hash,
    document_key,
    document_text,
    group_by_document,
)

logger = logging.getLogger(__name__)

# About 1200 tokens: Microsoft GraphRAG's default unit; most stories fit in one.
DEFAULT_UNIT_CHARS = 4000
DEFAULT_MIN_COMMUNITY_SIZE = 3
# A report prompt lists at most this many entities and relationships (by degree, weight).
MAX_REPORT_ENTITIES = 40
MAX_REPORT_RELATIONSHIPS = 60
# Relationships shown in one entity's profile at query time.
MAX_PROFILE_RELATIONSHIPS = 8
LOUVAIN_SEED = 42

EXTRACTION_PROMPT = """You extract a knowledge graph for a search index / Bạn trích xuất đồ thị tri thức cho chỉ mục tìm kiếm.

Rules / Quy tắc:
1. Use ONLY the text below / Chỉ dùng văn bản dưới đây
2. Entities: characters, creatures and deities, places, objects (magical ones especially), events / Thực thể: nhân vật, sinh vật và thần linh, địa điểm, đồ vật (nhất là vật thần kỳ), sự kiện
3. Name each entity by its shortest usual name, the same way every time / Gọi mỗi thực thể bằng tên ngắn quen dùng nhất, lần nào cũng giống nhau
4. Description: who or what it is and what it does in the text, at most 2 sentences / Mô tả: là ai, là gì và làm gì trong văn bản, tối đa 2 câu
5. Relationships only between listed entities, one sentence each / Quan hệ chỉ giữa các thực thể đã liệt kê, mỗi quan hệ một câu
6. Write in the text's language; return ONLY this JSON / Viết bằng ngôn ngữ của văn bản; chỉ trả về JSON này:
{{"entities": [{{"name": "...", "type": "person|creature|place|object|event|other", "description": "..."}}],
 "relationships": [{{"source": "...", "target": "...", "description": "..."}}]}}

Document / Tài liệu: {name}{part}
{text}"""

REPORT_PROMPT = """You write a community report for a search index / Bạn viết báo cáo cho một nhóm thực thể trong chỉ mục tìm kiếm.
These entities are closely connected in a knowledge graph built from the documents listed. The report answers questions about the group as a whole: what links them, recurring themes, how they appear across documents.
Các thực thể này liên kết chặt trong đồ thị tri thức dựng từ các tài liệu được liệt kê. Báo cáo dùng để trả lời câu hỏi về cả nhóm: điều gì nối chúng, chủ đề lặp lại, chúng xuất hiện thế nào trong các tài liệu.

Rules / Quy tắc:
1. Use ONLY the entities and relationships below / Chỉ dùng các thực thể và quan hệ dưới đây
2. Write in the language of the descriptions / Viết bằng ngôn ngữ của phần mô tả
3. At most 200 words, no other text / Tối đa 200 từ, không thêm gì khác:
Title / Tiêu đề:
Summary / Tóm tắt:
Key findings / Điểm chính: (2-5 short lines, each naming its documents / 2–5 dòng ngắn, mỗi dòng nêu tài liệu)

Documents / Tài liệu: {documents}

Entities / Thực thể:
{entities}

Relationships / Quan hệ:
{relationships}"""


def entity_key(name: str) -> str:
    """Merge key of an entity name: "Thạch  Sanh" and "thạch sanh" are one entity."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", name)).strip().casefold()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip()


def _join(first: str, second: str) -> str:
    if not first or second in first:
        return first or second
    return f"{first} {second}"


def _fold(text: str) -> str:
    from src.core.retriever import fold_vietnamese

    return " ".join(re.findall(r"[a-z0-9]+", fold_vietnamese(text)))


def display_name(metadata: Dict[str, Any]) -> str:
    """How prompts name a document: its file name, never its path (stable prompts)."""
    return str(metadata.get("relative_source") or metadata.get("file_name") or document_key(metadata) or "document")


def extraction_units(text: str, max_chars: int = DEFAULT_UNIT_CHARS) -> List[str]:
    """The text in slices of at most ``max_chars`` characters, cut between paragraphs."""
    units: List[str] = []
    current = ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        while len(paragraph) > max_chars:  # one huge paragraph: cut it
            if current:
                units.append(current)
                current = ""
            units.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if current and len(current) + 2 + len(paragraph) > max_chars:
            units.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        units.append(current)
    return units


def extraction_prompt(name: str, unit: str, index: int = 0, total: int = 1) -> str:
    part = f" (part / phần {index + 1}/{total})" if total > 1 else ""
    return EXTRACTION_PROMPT.format(name=name, part=part, text=unit)


def parse_extraction(text: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Entities and relationships of an extraction answer; None when it is not that JSON."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("entities", []), list):
        return None
    entities = [
        item for item in data.get("entities") or []
        if isinstance(item, dict) and _clean(item.get("name"))
    ]
    relationships = [
        item for item in data.get("relationships") or []
        if isinstance(item, dict) and _clean(item.get("source")) and _clean(item.get("target"))
    ]
    return {"entities": entities, "relationships": relationships}


def profile_id(document: str, key: str) -> str:
    return hashlib.sha1(f"graph-entity|{document}|{key}".encode("utf-8")).hexdigest()[:20]


@dataclass
class DocumentGraph:
    """Entities and relationships extracted from one document."""

    key: str
    name: str
    metadata: Dict[str, Any]
    content_sha256: str
    model: Optional[str] = None
    # entity key -> {"name", "type", "description"}
    entities: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # (source key, target key) -> description
    relationships: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def add_extraction(self, extraction: Dict[str, List[Dict[str, Any]]]) -> None:
        """Merge one extraction unit; the same name in two units is one entity."""
        for item in extraction["entities"]:
            name = _clean(item.get("name"))
            known = self.entities.get(entity_key(name))
            description = _clean(item.get("description"))
            if known is None:
                self.entities[entity_key(name)] = {
                    "name": name,
                    "type": _clean(item.get("type")).lower() or "other",
                    "description": description,
                }
            else:
                known["description"] = _join(known["description"], description)
        for item in extraction["relationships"]:
            source, target = _clean(item.get("source")), _clean(item.get("target"))
            source_key, target_key = entity_key(source), entity_key(target)
            if source_key == target_key:
                continue
            for key, name in ((source_key, source), (target_key, target)):
                self.entities.setdefault(key, {"name": name, "type": "other", "description": ""})
            pair = (source_key, target_key)
            self.relationships[pair] = _join(self.relationships.get(pair, ""), _clean(item.get("description")))

    def profiles(self) -> List[Document]:
        """One searchable profile per entity, with this document's metadata."""
        profiles = []
        for key in sorted(self.entities):
            entity = self.entities[key]
            outgoing = [
                [target, self.entities[target]["name"], description]
                for (source, target), description in sorted(self.relationships.items())
                if source == key
            ]
            lines = [f"{entity['name']} ({entity['type']}): {entity['description']}".rstrip(": ")]
            for (source, target), description in sorted(self.relationships.items()):
                if key in (source, target):
                    other = self.entities[target if source == key else source]["name"]
                    lines.append(f"- {entity['name']} — {other}: {description}".rstrip(": "))
            profiles.append(Document(
                page_content="\n".join(lines),
                metadata={
                    **self.metadata,
                    "graph_entity": key,
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "entity_description": entity["description"],
                    "graph_relations": json.dumps(outgoing, ensure_ascii=False),
                    "document_key": self.key,
                    "graph_document": self.name,
                    "content_sha256": self.content_sha256,
                    "graph_model": self.model,
                    "chunk_id": profile_id(self.key, key),
                },
            ))
        return profiles

    @classmethod
    def from_profiles(cls, profiles: Iterable[Document]) -> Dict[str, "DocumentGraph"]:
        """Document graphs rebuilt from stored profiles (after a restart)."""
        graphs: Dict[str, DocumentGraph] = {}
        for profile in profiles:
            metadata = profile.metadata or {}
            key = metadata.get("graph_entity")
            document = metadata.get("document_key")
            if not key or not document:
                continue
            graph = graphs.get(document)
            if graph is None:
                kept = {
                    name: value for name, value in metadata.items()
                    if not name.startswith(("graph_", "entity_")) and name not in (
                        "document_key", "content_sha256", "chunk_id", "relevance_score")
                }
                graph = graphs[document] = cls(
                    key=document,
                    name=metadata.get("graph_document") or display_name(kept),
                    metadata=kept,
                    content_sha256=metadata.get("content_sha256") or "",
                    model=metadata.get("graph_model"),
                )
            graph.entities[key] = {
                "name": metadata.get("entity_name") or key,
                "type": metadata.get("entity_type") or "other",
                "description": metadata.get("entity_description") or "",
            }
            try:
                relations = json.loads(metadata.get("graph_relations") or "[]")
            except json.JSONDecodeError:
                relations = []
            for target, target_name, description in relations:
                graph.entities.setdefault(target, {"name": target_name, "type": "other", "description": ""})
                graph.relationships[(key, target)] = description
        return graphs


@dataclass
class GraphCommunity:
    """Closely connected entities; ``id`` changes when anything its report reads changes."""

    id: str
    entities: List[str]
    documents: List[str]
    prompt: str


class GraphView:
    """The entity graph merged across documents."""

    def __init__(self, documents: Iterable[DocumentGraph]):
        self.documents = {graph.key: graph for graph in sorted(documents, key=lambda g: g.key)}
        # key -> {"name", "type", "descriptions": {document key: text}}
        self.entities: Dict[str, Dict[str, Any]] = {}
        # sorted key pair -> {"weight", "descriptions": {document key: text}}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for graph in self.documents.values():
            for key, entity in graph.entities.items():
                merged = self.entities.setdefault(
                    key, {"name": entity["name"], "type": entity["type"], "descriptions": {}})
                if merged["type"] == "other":
                    merged["type"] = entity["type"]
                merged["descriptions"][graph.key] = entity["description"]
            for (source, target), description in graph.relationships.items():
                edge = self.edges.setdefault(tuple(sorted((source, target))), {"weight": 0, "descriptions": {}})
                edge["weight"] += 1
                edge["descriptions"][graph.key] = _join(edge["descriptions"].get(graph.key, ""), description)
        self._folded_names: Optional[Dict[str, str]] = None
        self._adjacency: Dict[str, Set[str]] = {}
        for a, b in self.edges:
            self._adjacency.setdefault(a, set()).add(b)
            self._adjacency.setdefault(b, set()).add(a)

    def degree(self, key: str) -> int:
        return sum(self.edges[tuple(sorted((key, other)))]["weight"] for other in self._adjacency.get(key, ()))

    def neighbors(self, key: str) -> Set[str]:
        return self._adjacency.get(key, set())

    def communities(self, min_size: int = DEFAULT_MIN_COMMUNITY_SIZE, resolution: float = 1.0) -> List["GraphCommunity"]:
        """Louvain communities of at least ``min_size`` entities, each with its report prompt."""
        import networkx as nx

        if not self.edges:
            return []
        graph = nx.Graph()
        graph.add_nodes_from(sorted(self.entities))
        for (a, b), edge in sorted(self.edges.items()):
            graph.add_edge(a, b, weight=edge["weight"])
        groups = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=LOUVAIN_SEED)
        communities = []
        for members in groups:
            if len(members) < max(2, min_size):
                continue
            ordered = sorted(members, key=lambda key: (-self.degree(key), key))
            documents = sorted({doc for key in members for doc in self.entities[key]["descriptions"]})
            prompt = self.report_prompt(ordered, documents)
            communities.append(GraphCommunity(
                id=hashlib.sha1(f"graph-community|{prompt}".encode("utf-8")).hexdigest()[:20],
                entities=ordered,
                documents=documents,
                prompt=prompt,
            ))
        return sorted(communities, key=lambda community: community.id)

    def _document_names(self, documents: Iterable[str]) -> List[str]:
        return [self.documents[doc].name for doc in documents if doc in self.documents]

    def report_prompt(self, ordered: List[str], documents: List[str]) -> str:
        members = set(ordered)
        entity_lines = []
        for key in ordered[:MAX_REPORT_ENTITIES]:
            entity = self.entities[key]
            described = [
                f"{self.documents[doc].name}: {text}" for doc, text in sorted(entity["descriptions"].items()) if text
            ]
            entity_lines.append(f"- {entity['name']} ({entity['type']})" + (" — " + " | ".join(described) if described else ""))
        edges = sorted(
            ((pair, edge) for pair, edge in self.edges.items() if pair[0] in members and pair[1] in members),
            key=lambda item: (-item[1]["weight"], item[0]),
        )
        relationship_lines = [
            f"- {self.entities[a]['name']} — {self.entities[b]['name']}: "
            + " | ".join(f"{self.documents[doc].name}: {text}" for doc, text in sorted(edge["descriptions"].items()))
            for (a, b), edge in edges[:MAX_REPORT_RELATIONSHIPS]
        ]
        return REPORT_PROMPT.format(
            documents=", ".join(self._document_names(documents)),
            entities="\n".join(entity_lines),
            relationships="\n".join(relationship_lines) or "-",
        )

    def entity_profile(self, key: str, allowed: Optional[Set[str]] = None) -> Optional[Document]:
        """
        What the graph knows about one entity, for the LLM's context.

        ``allowed`` limits it to what those documents say (metadata filters).
        """
        entity = self.entities.get(key)
        if entity is None:
            return None
        documents = [doc for doc in sorted(entity["descriptions"]) if allowed is None or doc in allowed]
        if not documents:
            return None
        lines = [f"{entity['name']} ({entity['type']}; {', '.join(self._document_names(documents))})"]
        lines += [f"  {entity['descriptions'][doc]}" for doc in documents if entity["descriptions"][doc]]
        related = []
        for other in self.neighbors(key):
            edge = self.edges[tuple(sorted((key, other)))]
            texts = [text for doc, text in sorted(edge["descriptions"].items()) if allowed is None or doc in allowed]
            if texts:
                related.append((-edge["weight"], self.entities[other]["name"], " | ".join(texts)))
        for _, name, text in sorted(related)[:MAX_PROFILE_RELATIONSHIPS]:
            lines.append(f"  - {entity['name']} — {name}: {text}".rstrip(": "))
        return Document(
            page_content="\n".join(lines),
            metadata={
                "graph_entity": key,
                "entity_name": entity["name"],
                "documents": self._document_names(documents),
            },
        )

    def entities_named_in(self, text: str) -> List[str]:
        """Entities whose name appears in ``text``, with or without diacritics."""
        if self._folded_names is None:
            self._folded_names = {key: _fold(entity["name"]) for key, entity in self.entities.items()}
        folded = f" {_fold(text)} "
        found = [
            key for key, name in self._folded_names.items()
            if len(name) >= 3 and f" {name} " in folded
        ]
        return sorted(found, key=lambda key: (-len(key), key))


@dataclass
class GraphUpdate:
    """Graphs of newly loaded documents and the community reports they lead to."""

    graphs: Dict[str, DocumentGraph]
    reports: Dict[str, Document]


class GraphIndex:
    """
    Builds, stores and searches the knowledge graph of a document collection.

    Ingestion: ``prepare(docs)`` makes the LLM calls (outside the index lock),
    ``apply(update)`` stores the result. ``drop`` removes documents.
    """

    def __init__(
        self,
        entity_store,
        report_store,
        llm,
        unit_chars: int = DEFAULT_UNIT_CHARS,
        min_community_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
    ):
        try:
            import networkx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                'The knowledge graph needs networkx: pip install -e ".[graph]"'
            ) from e
        self.entity_store = entity_store
        self.report_store = report_store
        self.llm = llm
        self.unit_chars = unit_chars
        self.min_community_size = min_community_size
        self.documents: Dict[str, DocumentGraph] = {}
        self.reports: Dict[str, Document] = {}
        self.communities: List["GraphCommunity"] = []
        self.view = GraphView([])

    @property
    def num_entities(self) -> int:
        return len(self.view.entities)

    @property
    def num_relationships(self) -> int:
        return len(self.view.edges)

    # -- start-up -----------------------------------------------------------

    def restore(self) -> None:
        """Rebuild the graph from stored profiles and reports (no LLM call)."""
        try:
            self.documents = DocumentGraph.from_profiles(self.entity_store.get_all_documents())
            self.reports = {
                report.metadata["chunk_id"]: report
                for report in self.report_store.get_all_documents()
                if (report.metadata or {}).get("chunk_id")
            }
        except Exception as e:
            logger.warning(f"Could not load the knowledge graph: {e}")
            return
        self.view = GraphView(self.documents.values())
        self.communities = self.view.communities(self.min_community_size)
        missing = sum(1 for community in self.communities if community.id not in self.reports)
        if missing:
            logger.warning(f"{missing} graph communities have no report yet (written at the next ingest)")

    # -- ingestion ----------------------------------------------------------

    def prepare(self, all_docs: List[Document]) -> GraphUpdate:
        """LLM work for loaded documents: extraction, then reports of the resulting communities."""
        ask = _Asker(self.llm)
        graphs = self.extract(all_docs, ask)
        view = GraphView({**self.documents, **graphs}.values())
        reports = self._reports_for(view.communities(self.min_community_size), view, ask)
        return GraphUpdate(graphs, reports)

    def extract(self, all_docs: List[Document], ask: Optional["_Asker"] = None) -> Dict[str, DocumentGraph]:
        """
        Graph of each loaded document; one whose text did not change reuses its graph.

        A document whose extraction failed or was not valid JSON gets no new
        graph (it keeps the previous one, if any) and is extracted again the
        next time it is indexed.
        """
        ask = ask or _Asker(self.llm)
        graphs = {}
        for key, pages in group_by_document(all_docs).items():
            metadata = {
                name: value for name, value in (pages[0].metadata or {}).items()
                if name not in _NOT_DOCUMENT_LEVEL
            }
            sha = content_hash(pages)
            existing = self.documents.get(key)
            if existing is not None and existing.content_sha256 == sha:
                graphs[key] = DocumentGraph(
                    key=key, name=display_name(metadata), metadata=metadata, content_sha256=sha,
                    model=existing.model, entities=existing.entities, relationships=existing.relationships,
                )
                continue
            graph = DocumentGraph(
                key=key, name=display_name(metadata), metadata=metadata, content_sha256=sha,
                model=getattr(getattr(self.llm, "config", None), "model", None),
            )
            units = extraction_units(document_text(pages), self.unit_chars)
            complete = bool(units)
            for index, unit in enumerate(units):
                answer = ask(extraction_prompt(graph.name, unit, index, len(units)), key)
                extraction = parse_extraction(answer) if answer is not None else None
                if extraction is None:
                    if answer is not None:
                        logger.warning(f"Graph extraction for {key} was not valid JSON")
                    complete = False
                    continue
                graph.add_extraction(extraction)
            if complete:
                graphs[key] = graph
        ask.report()
        return graphs

    def _reports_for(
        self, communities: List["GraphCommunity"], view: GraphView, ask: Optional["_Asker"] = None
    ) -> Dict[str, Document]:
        """Reports of these communities: kept when unchanged, otherwise written by the LLM."""
        ask = ask or _Asker(self.llm)
        reports = {}
        for community in communities:
            report = self.reports.get(community.id)
            if report is None:
                text = ask(community.prompt, f"community {community.id}")
                if not text or not text.strip():
                    continue
                report = self._report_document(community, text, view)
            reports[community.id] = report
        ask.report()
        return reports

    def _report_document(self, community: GraphCommunity, text: str, view: GraphView) -> Document:
        return Document(
            page_content=text.strip(),
            metadata={
                "community_report": True,
                "chunk_id": community.id,
                "source": "knowledge graph community",
                "community_documents": json.dumps(community.documents, ensure_ascii=False),
                "community_entities": json.dumps(
                    [view.entities[key]["name"] for key in community.entities], ensure_ascii=False),
                "documents": ", ".join(view._document_names(community.documents)),
                "report_model": getattr(getattr(self.llm, "config", None), "model", None),
            },
        )

    def apply(self, update: GraphUpdate) -> None:
        """Store new document graphs, then recompute communities and their reports."""
        for key, graph in update.graphs.items():
            self.entity_store.delete(filter={"document_key": key})
            profiles = graph.profiles()
            if profiles:
                self.entity_store.add_documents(profiles, ids=[p.metadata["chunk_id"] for p in profiles])
            self.documents[key] = graph
        self.entity_store.persist()
        self._rebuild(update.reports)

    def drop(self, keep: Callable[[Document], bool], filter: Dict[str, Any]) -> None:
        """Remove the documents ``keep`` rejects (``filter`` matches their profiles)."""
        removed = [
            key for key, graph in self.documents.items()
            if not keep(Document(page_content="", metadata=graph.metadata))
        ]
        if not removed:
            return
        for key in removed:
            del self.documents[key]
        self.entity_store.delete(filter=filter)
        self.entity_store.persist()
        self._rebuild()

    def _rebuild(self, prepared: Optional[Dict[str, Document]] = None) -> None:
        """
        Communities of the current graph and their reports.

        Reports come from ``prepared`` (written before the index lock) or are
        kept from before; a community that has neither (documents removed
        meanwhile) is reported now.
        """
        self.view = GraphView(self.documents.values())
        self.communities = self.view.communities(self.min_community_size)
        known = {**self.reports, **(prepared or {})}
        missing = [community for community in self.communities if community.id not in known]
        written = self._reports_for(missing, self.view) if missing else {}
        reports = {
            community.id: written.get(community.id) or known.get(community.id)
            for community in self.communities
            if community.id in written or community.id in known
        }
        stale = [report_id for report_id in self.reports if report_id not in reports]
        new = [report for report_id, report in reports.items() if report_id not in self.reports]
        if stale:
            self.report_store.delete(ids=stale)
        if new:
            self.report_store.add_documents(new, ids=[report.metadata["chunk_id"] for report in new])
        if stale or new:
            self.report_store.persist()
        self.reports = reports

    # -- query --------------------------------------------------------------

    def allowed_documents(self, filter: Optional[Dict[str, Any]]) -> Optional[Set[str]]:
        """Documents a metadata filter lets through (None: no filter)."""
        if not filter:
            return None
        return {key for key, graph in self.documents.items() if metadata_matches(graph.metadata, filter)}

    def entity_candidates(self, query: str, filter: Optional[Dict[str, Any]], k: int) -> List[Document]:
        """
        Profiles of the entities a query may be about, merged across documents.

        Entities whose profile is close to the query come first, then those
        the query names (typed with or without diacritics).
        """
        if not self.view.entities or k < 1:
            return []
        keys: List[str] = []
        for hit in self.entity_store.similarity_search(query, k=2 * k, filter=filter):
            key = (hit.metadata or {}).get("graph_entity")
            if key and key not in keys:
                keys.append(key)
        for key in self.view.entities_named_in(query):
            if key not in keys:
                keys.append(key)
        allowed = self.allowed_documents(filter)
        profiles = [self.view.entity_profile(key, allowed) for key in keys[: 2 * k]]
        return [profile for profile in profiles if profile is not None]

    def report_candidates(self, query: str, filter: Optional[Dict[str, Any]], k: int) -> List[Document]:
        """
        Community reports close to the query.

        With a filter, a report qualifies only when every document it was
        built from passes the filter: a report mixes its documents.
        """
        if not self.reports or k < 1:
            return []
        allowed = self.allowed_documents(filter)
        hits = self.report_store.similarity_search(query, k=k if allowed is None else 3 * k)
        candidates = []
        for hit in hits:
            metadata = hit.metadata or {}
            if metadata.get("chunk_id") not in self.reports:
                continue
            if allowed is not None:
                documents = json.loads(metadata.get("community_documents") or "[]")
                if not set(documents) <= allowed:
                    continue
            candidates.append(hit)
        return candidates[:k]


class _Asker:
    """LLM calls of one batch; stops after several failed calls in a row (no key, provider down)."""

    def __init__(self, llm):
        self.llm = llm
        self.failures = 0
        self.skipped = 0

    def __call__(self, prompt: str, what: str) -> Optional[str]:
        if self.failures >= MAX_CONSECUTIVE_CARD_FAILURES:
            self.skipped += 1
            return None
        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            self.failures += 1
            logger.warning(f"Knowledge graph LLM call failed for {what}: {e}")
            return None
        self.failures = 0
        return answer

    def report(self) -> None:
        if self.skipped:
            logger.warning(
                f"{self.failures} knowledge graph LLM calls failed in a row: "
                f"{self.skipped} more skipped (done when the documents are indexed again)"
            )
            self.skipped = 0

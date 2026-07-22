"""
Library Manager Module
======================

Smart Knowledge Library & Automatic Document Organizer for RAG.

Features:
- Automatic categorization (Cổ tích/Kịch bản, Báo cáo/Tài chính, Học tập/Nghiên cứu, Vận hành/MMO, OCR Scan, Chung)
- Rich metadata tagging & summarization
- Structured folder organization (library/<category>/<file>)
- Persistent manifest tracker (.rag_library_manifest.json)
- Instant search & category filtering

Usage:
    from src.core.library_manager import LibraryManager

    lib = LibraryManager("D:/RAG/library")
    record = lib.ingest_and_organize("path/to/script.md")
    print(record["category"], record["library_path"])
"""

import json
import os
import re
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from langchain_core.documents import Document
from src.core.document_loader import DocumentLoader
from src.core.ocr_engine import OCREngine


class DocumentClassifier:
    """
    Automatic document classifier based on keyword scoring and content patterns.
    """

    CATEGORY_PATTERNS = {
        "cổ_tích_kịch_bản": [
            r"truyện", r"cổ tích", r"kịch bản", r"nhân vật", r"bối cảnh", r"scene",
            r"prompt", r"họa sĩ", r"cổ phong", r"sự tích", r"thần thoại", r"thạch sanh",
            r"tấm cám", r"trầu cau", r"guzheng", r"đàn tranh", r"narrative", r"character"
        ],
        "báo_cáo_tài_chính": [
            r"báo cáo", r"doanh thu", r"lợi nhuận", r"tài chính", r"ngân sách",
            r"chi phí", r"kế toán", r"doanh nghiệp", r"bảng kê", r"revenue", r"profit",
            r"balance sheet", r"income", r"financial"
        ],
        "học_tập_nghiên_cứu": [
            r"nghiên cứu", r"học thuật", r"luận văn", r"giáo trình", r"bài báo",
            r"thuật toán", r"kiến trúc", r"machine learning", r"deep learning",
            r"rag", r"embedding", r"model", r"research", r"paper", r"dataset"
        ],
        "vận_hành_mmo": [
            r"mmo", r"youtube", r"facebook", r"tiktok", r"seo", r"affiliate",
            r"vận hành", r"marketing", r"kênh", r"chiến dịch", r"view", r"subscriber",
            r"hướng dẫn đăng bài", r"chỉ số", r"ads"
        ],
        "tài_liệu_scan_ocr": [
            r"ocr", r"scan", r"ảnh scan", r"hóa đơn scan", r"sách scan", r"chữ viết tay"
        ]
    }

    @classmethod
    def classify(cls, text: str, file_type: str = "", filename: str = "") -> Tuple[str, List[str]]:
        """
        Classify document text into category and extract relevant tags.

        Args:
            text: Text content of the document
            file_type: Document file type
            filename: File name

        Returns:
            Tuple of (category_name, tags_list)
        """
        combined = f"{filename} {file_type} {text[:4000]}".lower()

        scores = {cat: 0 for cat in cls.CATEGORY_PATTERNS}

        for category, patterns in cls.CATEGORY_PATTERNS.items():
            for pat in patterns:
                matches = len(re.findall(r"\b" + pat + r"\b", combined))
                scores[category] += matches

        # OCR special rule if image format
        if file_type in ["image", "ocr_scan"] or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            scores["tài_liệu_scan_ocr"] += 5

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score == 0:
            category = "chung"
        else:
            category = best_category

        # Extract auto tags
        tags = []
        if "vietnamese" in combined or "tiếng việt" in combined:
            tags.append("Tiếng Việt")
        if "cổ tích" in combined or "fairy tale" in combined:
            tags.append("Cổ Tích")
        if "prompt" in combined:
            tags.append("Prompt AI")
        if file_type:
            tags.append(file_type.upper())

        return category, tags


class LibraryManager:
    """
    Smart Knowledge Library Manager for indexing, organizing, and querying documents.
    """

    def __init__(self, library_dir: Union[str, Path] = "library"):
        """
        Initialize Library Manager.

        Args:
            library_dir: Base directory for the Knowledge Library
        """
        self.library_dir = Path(library_dir).resolve()
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.library_dir / ".rag_library_manifest.json"
        
        self.loader = DocumentLoader()
        self.ocr_engine = OCREngine()
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest JSON file."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading library manifest: {e}")
        return {"documents": {}, "updated_at": None, "total_files": 0}

    def _save_manifest(self):
        """Save current manifest state to disk."""
        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["total_files"] = len(self.manifest.get("documents", {}))
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def ingest_and_organize(
        self,
        file_path: Union[str, Path],
        override_category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Ingest a file, perform OCR if needed, auto-classify, organize into category folder, and track in manifest.

        Args:
            file_path: Path to source file
            override_category: Optional explicit category
            extra_metadata: Extra metadata dictionary

        Returns:
            Tuple of (List[Document], record_dict)
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = self._compute_sha256(file_path)
        file_ext = file_path.suffix.lower()

        # 1. OCR text extraction for image files
        ocr_applied = False
        text_content = ""
        
        if file_ext in OCREngine.SUPPORTED_IMAGE_EXTENSIONS:
            text_content, _ = self.ocr_engine.extract_text_from_image(file_path)
            ocr_applied = True
            docs = [
                Document(
                    page_content=text_content or f"[Image File: {file_path.name}]",
                    metadata={
                        "source": str(file_path),
                        "file_type": "ocr_scan",
                        "file_name": file_path.name,
                        "sha256": sha256
                    }
                )
            ]
        else:
            # Standard loader
            docs = self.loader.load(file_path)
            text_content = "\n".join(d.page_content for d in docs)

        # 2. Document Classification
        file_type = docs[0].metadata.get("file_type", file_ext.replace(".", "")) if docs else file_ext
        category, auto_tags = DocumentClassifier.classify(text_content, file_type=file_type, filename=file_path.name)
        
        if override_category:
            category = override_category

        # 3. Organize File into Library Directory Structure (library/<category>/<filename>)
        category_dir = self.library_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = category_dir / file_path.name
        if file_path != target_path:
            shutil.copy2(file_path, target_path)

        # 4. Generate Summary & Rich Metadata
        summary = text_content[:200].replace("\n", " ").strip() + "..." if len(text_content) > 200 else text_content
        doc_id = sha256[:16]

        record = {
            "document_id": doc_id,
            "original_name": file_path.name,
            "library_path": str(target_path),
            "category": category,
            "file_type": file_type,
            "file_size": file_path.stat().st_size,
            "sha256": sha256,
            "ocr_applied": ocr_applied,
            "summary": summary,
            "tags": auto_tags,
            "added_at": datetime.now(timezone.utc).isoformat()
        }

        if extra_metadata:
            record.update(extra_metadata)

        # Attach library metadata to Document objects
        for doc in docs:
            doc.metadata["library_category"] = category
            doc.metadata["library_path"] = str(target_path)
            doc.metadata["document_id"] = doc_id
            doc.metadata["tags"] = auto_tags

        # Update manifest
        self.manifest["documents"][doc_id] = record
        self._save_manifest()

        logger.info(f"Ingested and organized '{file_path.name}' -> Category: [{category}]")
        return docs, record

    def list_library_categories(self) -> Dict[str, int]:
        """Get counts of documents grouped by category."""
        counts = {}
        for doc in self.manifest.get("documents", {}).values():
            cat = doc.get("category", "chung")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_documents_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all library document records belonging to a category."""
        return [
            doc for doc in self.manifest.get("documents", {}).values()
            if doc.get("category") == category
        ]

    def search_library(self, query: str) -> List[Dict[str, Any]]:
        """Search manifest documents by filename, tags, summary, or category."""
        q = query.lower()
        results = []
        for doc in self.manifest.get("documents", {}).values():
            if (
                q in doc.get("original_name", "").lower()
                or q in doc.get("category", "").lower()
                or q in doc.get("summary", "").lower()
                or any(q in t.lower() for t in doc.get("tags", []))
            ):
                results.append(doc)
        return results

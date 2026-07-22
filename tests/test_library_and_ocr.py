"""
Tests for Smart Knowledge Library & OCR Engine
================================================
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.core.ocr_engine import OCREngine, clean_vietnamese_ocr_text
from src.core.library_manager import LibraryManager, DocumentClassifier


def test_clean_vietnamese_ocr_text():
    """Verify Vietnamese OCR text cleaner handles hyphenation and blank lines."""
    raw_ocr = "Thạch Sanh chiến đấu với đại-\nbằng cứu công chúa.\n\n\n\nKết thúc có hậu."
    cleaned = clean_vietnamese_ocr_text(raw_ocr)
    assert "đạibằng" in cleaned or "đại bằng" in cleaned
    assert "\n\n\n" not in cleaned


def test_document_classifier_categories():
    """Verify document classifier routes texts to correct library categories."""
    # Fairy tale script
    cat1, tags1 = DocumentClassifier.classify("Truyện cổ tích Thạch Sanh kịch bản nhân vật bối cảnh", file_type="markdown")
    assert cat1 == "cổ_tích_kịch_bản"

    # Financial report
    cat2, tags2 = DocumentClassifier.classify("Báo cáo tài chính doanh thu lợi nhuận kế toán năm 2026", file_type="pdf")
    assert cat2 == "báo_cáo_tài_chính"

    # Research paper
    cat3, tags3 = DocumentClassifier.classify("Nghiên cứu thuật toán RAG embedding vector search machine learning", file_type="pdf")
    assert cat3 == "học_tập_nghiên_cứu"

    # MMO guide
    cat4, tags4 = DocumentClassifier.classify("Quy trình vận hành MMO Youtube Facebook SEO tăng view subscriber", file_type="txt")
    assert cat4 == "vận_hành_mmo"


def test_library_manager_ingest_and_organize(tmp_path):
    """Verify LibraryManager ingests files, categorizes, and creates manifest."""
    lib_dir = tmp_path / "test_library"
    manager = LibraryManager(library_dir=lib_dir)

    # Create dummy source text file
    sample_file = tmp_path / "su_tich_co_tich.txt"
    sample_file.write_text("Truyện cổ tích sự tích trầu cau bối cảnh nhân vật kịch bản.", encoding="utf-8")

    docs, record = manager.ingest_and_organize(sample_file)

    assert len(docs) > 0
    assert record["category"] == "cổ_tích_kịch_bản"
    assert Path(record["library_path"]).exists()
    assert (lib_dir / "cổ_tích_kịch_bản" / "su_tich_co_tich.txt").exists()
    assert manager.manifest_path.exists()

    # Test search library
    results = manager.search_library("cổ tích")
    assert len(results) >= 1
    assert results[0]["original_name"] == "su_tich_co_tich.txt"


def test_naive_rag_ingest_to_library(tmp_path):
    """Verify NaiveRAG ingest_to_library workflow."""
    from src.rag.naive_rag import NaiveRAG

    rag = NaiveRAG.__new__(NaiveRAG)
    rag.document_loader = Mock()
    rag.text_splitter = Mock()
    rag.embeddings = Mock()
    rag.vector_store = Mock()
    rag.llm = Mock()
    rag.retrieval_k = 4
    rag.system_prompt = "Prompt"
    rag._documents = []
    rag._chunks = []
    
    lib_dir = tmp_path / "rag_lib"
    rag.library_manager = LibraryManager(library_dir=lib_dir)

    sample_doc = tmp_path / "bao_cao_doanh_thu.txt"
    sample_doc.write_text("Báo cáo tài chính doanh thu lợi nhuận tháng 7 năm 2026.", encoding="utf-8")

    rag.text_splitter.split_documents.side_effect = lambda docs: docs

    result = rag.ingest_to_library(sample_doc)
    assert result["status"] == "success"
    assert result["files_processed"] == 1
    assert (lib_dir / "báo_cáo_tài_chính" / "bao_cao_doanh_thu.txt").exists()

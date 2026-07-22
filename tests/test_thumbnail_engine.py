"""
Test for Fairy Tale Thumbnail & Master Prompt Engine
"""

import pytest
from pathlib import Path
from src.story.thumbnail_engine import FairyTaleThumbnailEngine


def test_thumbnail_engine_generates_master_prompt():
    engine = FairyTaleThumbnailEngine()
    prompt = engine.generate_master_thumbnail_prompt({
        "title_main": "TRẦU CAU",
        "title_prefix": "SỰ TÍCH",
        "category_badge": "CỔ TÍCH VIỆT NAM"
    })
    
    assert "Sự Tích Trầu Cau" in prompt
    assert "TRẦU CAU" in prompt
    assert "=== SCENE COMPOSITION ===" in prompt
    assert "=== TEXT LAYOUT" in prompt
    assert "=== CTR OPTIMIZATION NOTES ===" in prompt


def test_thumbnail_engine_export(tmp_path):
    engine = FairyTaleThumbnailEngine()
    out = tmp_path / "master_prompt.txt"
    filepath = engine.export_prompt({
        "title_main": "TRẦU CAU",
        "title_prefix": "SỰ TÍCH"
    }, out)
    
    assert Path(filepath).exists()
    content = Path(filepath).read_text(encoding="utf-8")
    assert "SCENE COMPOSITION" in content

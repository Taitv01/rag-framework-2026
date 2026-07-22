"""
OCR Engine Module
=================

Advanced OCR and image text conversion engine supporting Vietnamese and English.
Provides fallback mechanisms for scanned PDFs, images, and documents.

Supported image formats:
- PNG, JPG, JPEG, WEBP, BMP, TIFF

Usage:
    from src.core.ocr_engine import OCREngine

    ocr = OCREngine()
    text, confidence = ocr.extract_text_from_image("scanned_doc.png")
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


def clean_vietnamese_ocr_text(text: str) -> str:
    """
    Clean and fix common OCR artifacts in Vietnamese text.

    Args:
        text: Raw OCR extracted text

    Returns:
        Cleaned text with normalized Vietnamese characters
    """
    if not text:
        return ""

    # Fix line break hyphenations (e.g., "thư- \nvẫn" -> "thư vẫn")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    
    # Fix multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize weird unicode spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")

    # Clean leading/trailing spaces on lines
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


class OCREngine:
    """
    Engine for OCR text extraction from images and scanned documents.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, lang: str = "vie+eng"):
        """
        Initialize OCR Engine.

        Args:
            lang: Languages to use for OCR (default: Vietnamese + English)
        """
        self.lang = lang
        self.has_pytesseract = False
        self.has_easyocr = False
        self.easyocr_reader = None

        # Check pytesseract availability
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.has_pytesseract = True
            logger.info("PyTesseract OCR available.")
        except ImportError:
            logger.debug("PyTesseract not installed.")

        # Check EasyOCR fallback
        try:
            import easyocr
            self.easyocr = easyocr
            self.has_easyocr = True
            logger.info("EasyOCR available.")
        except ImportError:
            logger.debug("EasyOCR not installed.")

    def is_available(self) -> bool:
        """Return True if at least one OCR backend is available."""
        return self.has_pytesseract or self.has_easyocr

    def extract_text_from_image(self, image_input: Union[str, Path, Any]) -> Tuple[str, float]:
        """
        Extract text from an image path or PIL Image object.

        Args:
            image_input: File path or PIL Image instance

        Returns:
            Tuple of (extracted_text, confidence_score_between_0_and_1)
        """
        try:
            from PIL import Image
        except ImportError:
            logger.error("Pillow (PIL) library is required for image processing.")
            return "", 0.0

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                logger.error(f"Image file not found: {img_path}")
                return "", 0.0
            image = Image.open(img_path)
        else:
            image = image_input

        # 1. Try PyTesseract
        if self.has_pytesseract:
            try:
                extracted = self.pytesseract.image_to_string(image, lang=self.lang)
                cleaned = clean_vietnamese_ocr_text(extracted)
                if cleaned:
                    return cleaned, 0.90
            except Exception as e:
                logger.warning(f"PyTesseract extraction failed: {e}. Trying fallback...")

        # 2. Try EasyOCR
        if self.has_easyocr:
            try:
                if self.easyocr_reader is None:
                    self.easyocr_reader = self.easyocr.Reader(['vi', 'en'], gpu=False)
                import numpy as np
                img_np = np.array(image)
                results = self.easyocr_reader.readtext(img_np)
                lines = [res[1] for res in results]
                cleaned = clean_vietnamese_ocr_text("\n".join(lines))
                if cleaned:
                    return cleaned, 0.85
            except Exception as e:
                logger.warning(f"EasyOCR extraction failed: {e}")

        # Fallback message if no OCR engine succeeded
        logger.warning("No active OCR backend returned text.")
        return "", 0.0

    def extract_text_from_pdf_pages(self, pdf_path: Union[str, Path]) -> Tuple[str, int]:
        """
        Perform OCR on scanned PDF pages.

        Args:
            pdf_path: Path to scanned PDF file

        Returns:
            Tuple of (extracted_combined_text, pages_processed_count)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return "", 0

        text_pages = []
        
        # Try pdf2image if available
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path)
            for idx, img in enumerate(images, 1):
                txt, _ = self.extract_text_from_image(img)
                if txt:
                    text_pages.append(f"--- [Page {idx}] ---\n{txt}")
            return "\n\n".join(text_pages), len(images)
        except Exception as e:
            logger.debug(f"pdf2image fallback skipped: {e}")

        return "", 0

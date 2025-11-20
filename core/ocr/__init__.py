"""
OCR Module
Optical Character Recognition for extracting text from document images.
"""

from .ocr_engine import extract_text_from_image, batch_ocr_text_blocks
from .postprocess_text import clean_ocr_output, merge_text_lines, format_as_markdown

__all__ = [
    'extract_text_from_image',
    'batch_ocr_text_blocks',
    'clean_ocr_output',
    'merge_text_lines',
    'format_as_markdown'
]

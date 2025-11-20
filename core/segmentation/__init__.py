"""
Segmentation Module
Document layout analysis and region segmentation for text and figures.
"""

from .segmenter import segment_page_into_regions, extract_text_blocks, extract_figure_blocks
from .morphology_utils import apply_dilation, apply_erosion, apply_opening, apply_closing

__all__ = [
    'segment_page_into_regions',
    'extract_text_blocks',
    'extract_figure_blocks',
    'apply_dilation',
    'apply_erosion',
    'apply_opening',
    'apply_closing'
]

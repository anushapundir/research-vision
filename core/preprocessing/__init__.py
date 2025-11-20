"""
Preprocessing Module
Classical image processing techniques for document enhancement.
"""

from .preprocess import (
    to_grayscale,
    equalize_histogram,
    equalize_clahe,
    denoise_gaussian,
    denoise_median,
    binarize_otsu,
    binarize_adaptive,
    run_full_preprocessing
)

__all__ = [
    'to_grayscale',
    'equalize_histogram',
    'equalize_clahe',
    'denoise_gaussian',
    'denoise_median',
    'binarize_otsu',
    'binarize_adaptive',
    'run_full_preprocessing'
]

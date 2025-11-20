"""
Preprocessing Module
Implements classical Image & Video Processing techniques for document enhancement.
Includes: grayscale conversion, histogram equalization, denoising, and binarization.
"""

import cv2
import numpy as np
from typing import Dict


def to_grayscale(page: np.ndarray) -> np.ndarray:
    """
    Convert a color image to grayscale.
    
    Args:
        page (np.ndarray): Input image in BGR format
        
    Returns:
        np.ndarray: Grayscale image
        
    Note:
        Uses weighted average: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(page.shape) == 2:
        # Already grayscale
        return page
    return cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        gray (np.ndarray): Grayscale input image
        
    Returns:
        np.ndarray: Contrast-enhanced image
        
    Note:
        Redistributes pixel intensities to use the full dynamic range [0, 255].
        Improves visibility of details in low-contrast regions.
    """
    return cv2.equalizeHist(gray)


def equalize_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        gray (np.ndarray): Grayscale input image
        clip_limit (float): Threshold for contrast limiting
        tile_size (int): Size of grid for histogram equalization
        
    Returns:
        np.ndarray: Adaptively contrast-enhanced image
        
    Note:
        CLAHE is better than standard histogram equalization as it avoids
        over-amplification of noise in homogeneous regions.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def denoise_gaussian(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        gray (np.ndarray): Grayscale input image
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        
    Returns:
        np.ndarray: Blurred image
        
    Note:
        Gaussian blur uses a weighted average where nearby pixels contribute more.
        Good for removing Gaussian noise while preserving edges reasonably well.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def denoise_median(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median blur to reduce salt-and-pepper noise.
    
    Args:
        gray (np.ndarray): Grayscale input image
        kernel_size (int): Size of the median filter kernel (must be odd)
        
    Returns:
        np.ndarray: Denoised image
        
    Note:
        Median filter is excellent for removing salt-and-pepper noise
        while preserving edges better than Gaussian blur.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    return cv2.medianBlur(gray, kernel_size)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to create a binary image.
    
    Args:
        gray (np.ndarray): Grayscale input image
        
    Returns:
        np.ndarray: Binary image (0 or 255)
        
    Note:
        Otsu's method automatically finds the optimal threshold value
        by minimizing intra-class variance. Works well when the histogram
        has a bimodal distribution (foreground and background).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to create a binary image.
    
    Args:
        gray (np.ndarray): Grayscale input image
        block_size (int): Size of pixel neighborhood for threshold calculation (must be odd)
        C (int): Constant subtracted from the mean
        
    Returns:
        np.ndarray: Binary image (0 or 255)
        
    Note:
        Adaptive thresholding calculates threshold for local regions.
        Better than Otsu for images with varying illumination conditions.
        Uses Gaussian-weighted average of neighborhood.
    """
    if block_size % 2 == 0:
        block_size += 1  # Ensure odd block size
    return cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        C
    )


def run_full_preprocessing(page: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run the complete preprocessing pipeline on a page image.
    
    Args:
        page (np.ndarray): Input page image (BGR or grayscale)
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing all preprocessing stages:
            - 'original': Original input image
            - 'gray': Grayscale conversion
            - 'equalized': Histogram equalized
            - 'clahe': CLAHE enhanced
            - 'gaussian': Gaussian blur applied
            - 'median': Median blur applied
            - 'otsu': Otsu thresholded binary
            - 'adaptive': Adaptive thresholded binary
            
    Example:
        >>> results = run_full_preprocessing(page_image)
        >>> cv2.imshow("Original", results['original'])
        >>> cv2.imshow("Binary", results['otsu'])
    """
    results = {}
    
    # Store original
    results['original'] = page.copy()
    
    # Step 1: Convert to grayscale
    gray = to_grayscale(page)
    results['gray'] = gray
    
    # Step 2: Histogram equalization for contrast enhancement
    equalized = equalize_histogram(gray)
    results['equalized'] = equalized
    
    # Step 2b: CLAHE (alternative adaptive equalization)
    clahe = equalize_clahe(gray)
    results['clahe'] = clahe
    
    # Step 3: Denoising with Gaussian blur
    gaussian = denoise_gaussian(equalized, kernel_size=5)
    results['gaussian'] = gaussian
    
    # Step 4: Denoising with Median blur
    median = denoise_median(equalized, kernel_size=3)
    results['median'] = median
    
    # Step 5: Binarization using Otsu's method
    otsu = binarize_otsu(gaussian)
    results['otsu'] = otsu
    
    # Step 6: Binarization using Adaptive thresholding
    adaptive = binarize_adaptive(median, block_size=11, C=2)
    results['adaptive'] = adaptive
    
    return results


if __name__ == "__main__":
    # Quick test
    print("Preprocessing module loaded successfully")
    print("Available functions:")
    print("  - to_grayscale()")
    print("  - equalize_histogram()")
    print("  - equalize_clahe()")
    print("  - denoise_gaussian()")
    print("  - denoise_median()")
    print("  - binarize_otsu()")
    print("  - binarize_adaptive()")
    print("  - run_full_preprocessing()")

"""
PDF Loader Module
Converts PDF pages to OpenCV-compatible numpy arrays using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
import numpy as np
from typing import List
from PIL import Image
import io


def load_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """
    Load a PDF file and convert each page to a numpy array image.
    
    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Resolution for rendering pages (default: 200)
        
    Returns:
        List[np.ndarray]: List of page images as numpy arrays in BGR format (OpenCV compatible)
        
    Example:
        >>> pages = load_pdf_to_images("paper.pdf", dpi=300)
        >>> print(f"Loaded {len(pages)} pages")
    """
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        pages = []
        
        # Calculate zoom factor based on DPI
        # PyMuPDF default is 72 DPI, so zoom = target_dpi / 72
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Convert each page to an image
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            
            # Convert RGB to BGR (OpenCV format)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1]  # RGB to BGR
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA to BGR
                img_array = img_array[:, :, [2, 1, 0]]
            
            pages.append(img_array)
        
        pdf_document.close()
        return pages
        
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")


def save_page_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a page image to disk.
    
    Args:
        image (np.ndarray): Image array in BGR format
        output_path (str): Path where the image will be saved
    """
    import cv2
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    # Quick test
    print("PDF Loader module loaded successfully")
    print("Use load_pdf_to_images(pdf_path, dpi=200) to convert PDF to images")

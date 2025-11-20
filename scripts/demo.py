"""
Research Vision - CLI Demo Script
Tests the preprocessing pipeline without a UI.
Loads a PDF, processes all pages, and saves intermediate outputs.

Usage:
    python scripts/demo.py

Instructions:
    1. Place a PDF file in the 'data/' folder (or update PDF_PATH below)
    2. Run: python scripts/demo.py
    3. Check 'outputs/pages/' and 'outputs/preprocessed/' for the generated images
"""

import os
import sys
import cv2
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from scripts/ to project root
sys.path.insert(0, str(PROJECT_ROOT))

# Updated imports for new structure
from core.pdf.pdf_loader import load_pdf_to_images
from core.preprocessing.preprocess import run_full_preprocessing
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PAGES_DIR = OUTPUTS_DIR / "pages"
PREPROCESSED_DIR = OUTPUTS_DIR / "preprocessed"

# Default PDF path - change this to your PDF
PDF_PATH = DATA_DIR / "attention is all you need.pdf"
DPI = 200  # Resolution for PDF rendering
PROCESS_ALL_PAGES = True  # Set to False to process only first page


def main():
    """
    Main function to test the preprocessing pipeline.
    """
    print("=" * 60)
    print("Research Vision - PDF Preprocessing")
    print("=" * 60)
    
    # Check if PDF exists
    if not PDF_PATH.exists():
        print(f"\n❌ ERROR: PDF file not found at '{PDF_PATH}'")
        print(f"Please place a PDF file in the 'data/' folder or update PDF_PATH in this script.")
        
        # List available PDFs
        if DATA_DIR.exists():
            pdfs = list(DATA_DIR.glob("*.pdf"))
            if pdfs:
                print(f"\n📁 Available PDF files in {DATA_DIR}:")
                for pdf in pdfs:
                    print(f"   - {pdf.name}")
        return
    
    # Create output directories
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load PDF to images
    print(f"\n📄 Loading PDF: {PDF_PATH.name}")
    print(f"   DPI: {DPI}")
    
    try:
        pages = load_pdf_to_images(str(PDF_PATH), dpi=DPI)
        print(f"✓ Successfully loaded {len(pages)} page(s)")
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return
    
    # Determine which pages to process
    pages_to_process = range(len(pages)) if PROCESS_ALL_PAGES else [0]
    
    print(f"\n🖼️  Processing {len(list(pages_to_process))} page(s)")
    
    # Step 2: Process pages
    total_saved = 0
    for page_idx in pages_to_process:
        page = pages[page_idx]
        print(f"\n⚙️  Processing page {page_idx + 1}/{len(pages)}...")
        print(f"   Image shape: {page.shape}")
        
        # Run preprocessing pipeline
        results = run_full_preprocessing(page)
        
        # Save original page to pages/ directory
        page_path = PAGES_DIR / f"page_{page_idx + 1:03d}.png"
        cv2.imwrite(str(page_path), results['original'])
        total_saved += 1
        
        # Save preprocessing results to preprocessed/ directory
        for name, image in results.items():
            if name == 'original':
                continue  # Already saved
            filename = f"page_{page_idx + 1:03d}_{name}.png"
            filepath = PREPROCESSED_DIR / filename
            cv2.imwrite(str(filepath), image)
            total_saved += 1
        
        print(f"   ✓ Saved {len(results)} outputs for page {page_idx + 1}")
    
    print("\n" + "=" * 60)
    print("✅ DONE! All preprocessing outputs saved successfully.")
    print("=" * 60)
    print(f"\nProcessed {len(list(pages_to_process))} page(s)")
    print(f"Saved {total_saved} images total")
    print(f"\n📂 Output directories:")
    print(f"   - Pages: {PAGES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"   - Preprocessed: {PREPROCESSED_DIR.relative_to(PROJECT_ROOT)}/")
    print("\n💡 Run the web app for interactive visualization:")
    print("   streamlit run app/streamlit_preprocess.py")
    print()


if __name__ == "__main__":
    main()

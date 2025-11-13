"""
Research Vision - CLI Demo Script
Tests the preprocessing pipeline without a UI.
Loads a PDF, processes all pages, and saves intermediate outputs.

Usage:
    python main_demo.py

Instructions:
    1. Place a PDF file in the 'data/' folder (or update PDF_PATH below)
    2. Run: python main_demo.py
    3. Check 'outputs/preprocess/' for the generated images
"""

import os
import cv2
from app.pdf_loader import load_pdf_to_images
from app.preprocess import run_full_preprocessing


# Configuration
PDF_PATH = "data/sample.pdf"  # Update this with your PDF path
DPI = 200  # Resolution for PDF rendering
OUTPUT_DIR = "outputs/preprocess"
PROCESS_ALL_PAGES = True  # Set to False to process only first page


def main():
    """
    Main function to test the preprocessing pipeline.
    """
    print("=" * 60)
    print("Research Vision - PDF Preprocessing")
    print("=" * 60)
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"\n❌ ERROR: PDF file not found at '{PDF_PATH}'")
        print(f"Please place a PDF file in the 'data/' folder or update PDF_PATH in this script.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load PDF to images
    print(f"\n📄 Loading PDF: {PDF_PATH}")
    print(f"   DPI: {DPI}")
    
    try:
        pages = load_pdf_to_images(PDF_PATH, dpi=DPI)
        print(f"✓ Successfully loaded {len(pages)} page(s)")
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return
    
    # Determine which pages to process
    pages_to_process = range(len(pages)) if PROCESS_ALL_PAGES else [0]
    
    print(f"\n🖼️  Processing {len(list(pages_to_process))} page(s)")
    
    # Step 2: Process pages
    total_saved = 0
    for page_idx in (range(len(pages)) if PROCESS_ALL_PAGES else [0]):
        page = pages[page_idx]
        print(f"\n⚙️  Processing page {page_idx + 1}/{len(pages)}...")
        print(f"   Image shape: {page.shape}")
        
        # Run preprocessing pipeline
        results = run_full_preprocessing(page)
        
        # Save all outputs
        for name, image in results.items():
            filename = f"page_{page_idx + 1:03d}_{name}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, image)
            total_saved += 1
        
        print(f"   ✓ Saved {len(results)} outputs for page {page_idx + 1}")
    
    print("\n" + "=" * 60)
    print("✅ DONE! All preprocessing outputs saved successfully.")
    print("=" * 60)
    print(f"\nProcessed {len(list(range(len(pages)) if PROCESS_ALL_PAGES else [0]))} page(s)")
    print(f"Saved {total_saved} images total")
    print(f"\n📂 Check the '{OUTPUT_DIR}/' folder to view results.")
    print("\n💡 Run the web app for interactive visualization:")
    print("   python -m streamlit run app_streamlit_preprocess.py")
    print()


if __name__ == "__main__":
    main()

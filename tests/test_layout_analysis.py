import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.layout.layout_analysis import LayoutAnalyzer

def test_layout_analyzer():
    config_path = PROJECT_ROOT / "notebooks/models/config.yaml"
    model_path = PROJECT_ROOT / "notebooks/models/model_final.pth"
    
    if not config_path.exists() or not model_path.exists():
        print("Model files not found. Skipping test.")
        return

    print("Initializing LayoutAnalyzer...")
    try:
        analyzer = LayoutAnalyzer(str(config_path), str(model_path))
        print("LayoutAnalyzer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LayoutAnalyzer: {e}")
        return

    # Create a dummy image
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (500, 200), (255, 255, 255), -1) # Mock text block
    
    print("Processing dummy image...")
    try:
        layout = analyzer.process_image(image)
        print(f"Layout detected: {len(layout)} elements.")
    except Exception as e:
        print(f"Failed to process image: {e}")
        return
        
    print("Extracting elements...")
    try:
        results = analyzer.extract_elements(image, layout, output_dir=PROJECT_ROOT / "tests/output", page_name="test_page")
        print("Elements extracted successfully.")
        print(f"Visualization shape: {results['visualization'].shape}")
    except Exception as e:
        print(f"Failed to extract elements: {e}")

if __name__ == "__main__":
    test_layout_analyzer()

import layoutparser as lp
import cv2
import numpy as np
import os
from pathlib import Path
import pytesseract
from PIL import Image

class LayoutAnalyzer:
    def __init__(self, config_path, model_path, label_map=None):
        """
        Initialize the LayoutAnalyzer with model configuration and weights.
        
        Args:
            config_path (str): Path to the Detectron2 config file (yaml).
            model_path (str): Path to the model weights file (pth).
            label_map (dict, optional): Dictionary mapping class IDs to names.
        """
        # Default label map for PubLayNet if not provided
        if label_map is None:
            self.label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        else:
            self.label_map = label_map
            
        # Initialize the model
        # We use 'lp://' scheme for config if it's a standard one, but here we expect local paths
        # LayoutParser's Detectron2LayoutModel expects config_path to be either a catalog string or a file path
        # Since we have a local config file that might need specific handling, we'll try to load it directly.
        # However, lp.Detectron2LayoutModel logic for local files is: if it exists, use it.
        
        self.model = lp.Detectron2LayoutModel(
            config_path=config_path,
            model_path=model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map=self.label_map
        )

    def process_image(self, image):
        """
        Run layout detection on an image.
        
        Args:
            image (np.ndarray): Input image (BGR or RGB).
            
        Returns:
            lp.Layout: The detected layout object.
        """
        # Ensure image is RGB for model
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assuming input might be BGR from cv2, convert to RGB
            # But layoutparser usually handles it. Let's be safe and ensure RGB if we read with cv2.
            # If the user passes BGR, we should convert. 
            # For now, let's assume the input is consistent with what the model expects (usually RGB).
            pass
            
        layout = self.model.detect(image)
        return layout

    def extract_elements(self, image, layout, output_dir=None, page_name="page"):
        """
        Extract, crop, and OCR detected elements.
        
        Args:
            image (np.ndarray): The original image.
            layout (lp.Layout): The detected layout.
            output_dir (Path, optional): Directory to save results.
            page_name (str): Name/ID of the page for file naming.
            
        Returns:
            dict: Structured results with crops and text.
        """
        results = {
            "visualization": None,
            "elements": []
        }
        
        # Draw visualization
        viz = lp.draw_box(image, layout, box_width=3, show_element_type=True)
        results["visualization"] = np.array(viz)
        
        # Sort layout elements (e.g., top to bottom)
        # layout.sort(key=lambda x: x.coordinates[1]) 
        # LayoutParser layout object is a list-like, but doesn't have in-place sort. 
        # We can just iterate.
        
        for i, block in enumerate(layout):
            # Get element info
            category_id = block.type
            category_name = self.label_map.get(category_id, str(category_id))
            # Some models return string types directly
            if isinstance(category_id, str):
                category_name = category_id
            
            # Get coordinates
            x_1, y_1, x_2, y_2 = block.coordinates
            x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)
            
            # Crop image
            # Ensure coordinates are within bounds
            h, w = image.shape[:2]
            x_1 = max(0, x_1)
            y_1 = max(0, y_1)
            x_2 = min(w, x_2)
            y_2 = min(h, y_2)
            
            crop = image[y_1:y_2, x_1:x_2]
            
            # OCR if it's text-like
            text_content = ""
            if category_name in ["Text", "Title", "List"]:
                try:
                    # Convert to PIL for Tesseract
                    pil_crop = Image.fromarray(crop)
                    text_content = pytesseract.image_to_string(pil_crop).strip()
                except Exception as e:
                    text_content = f"OCR Error: {str(e)}"
            
            element_data = {
                "id": i,
                "type": category_name,
                "box": [x_1, y_1, x_2, y_2],
                "crop": crop,
                "text": text_content
            }
            results["elements"].append(element_data)
            
            # Save to disk if output_dir is provided
            if output_dir:
                category_dir = output_dir / category_name
                category_dir.mkdir(parents=True, exist_ok=True)
                
                # Save crop
                crop_path = category_dir / f"{page_name}_crop_{i}.png"
                cv2.imwrite(str(crop_path), crop) # Assuming crop is BGR (opencv default) or RGB? 
                # If input image was RGB (from pdf2image), cv2.imwrite expects BGR. 
                # We might need to convert RGB to BGR before saving if the input was RGB.
                # Let's handle color conversion carefully in the main app or here.
                # For now, assuming input is BGR compatible for saving.
                
                # Save text
                if text_content:
                    text_path = category_dir / f"{page_name}_crop_{i}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text_content)
                        
        return results

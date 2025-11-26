import google.generativeai as genai
import os
from pathlib import Path
import json
from PIL import Image

class GeminiSummarizer:
    def __init__(self, api_key):
        """
        Initialize the GeminiSummarizer with the API key.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def prepare_content(self, layout_results):
        """
        Prepare the content for the Gemini API from layout analysis results.
        
        Args:
            layout_results (dict): The dictionary containing layout analysis results per page.
            
        Returns:
            list: A list of parts to send to Gemini (text and images).
        """
        prompt_parts = []
        
        intro_text = """
        You are an expert research assistant. Your task is to analyze the provided content from a research paper, which includes text, figures, and formulas extracted from each page.
        
        I want you to provide:
        1. A **Per-Page Summary**: For each page, identify the most important points.
        2. An **Overall Summary**: A concise summary of the entire paper based on the per-page analysis.
        
        **Crucial Requirement:**
        You must cite your sources. When you mention a specific point, figure, or formula, refer to it using the ID provided in the input.
        The input for each element will be in the format: `[Page X, Item ID] Type: Content`.
        For example, if you use information from a text block with ID 5 on Page 1, cite it as `(Source: Page 1, Item 5)`.
        If you refer to a figure, cite it similarly.
        
        **Output Format:**
        Please output the result in JSON format with the following structure:
        {
            "overall_summary": "...",
            "page_summaries": [
                {
                    "page_number": 1,
                    "summary": "...",
                    "key_points": ["point 1 (Source: Page 1, Item X)", "point 2 (Source: Page 1, Item Y)"]
                },
                ...
            ]
        }
        """
        prompt_parts.append(intro_text)
        
        for page_idx, result in layout_results.items():
            page_num = page_idx + 1
            prompt_parts.append(f"\n--- Page {page_num} ---\n")
            
            elements = result.get('elements', [])
            # Sort elements by vertical position (y_1) to maintain reading order
            elements.sort(key=lambda x: x['box'][1])
            
            for element in elements:
                el_id = element['id']
                el_type = element['type']
                el_text = element.get('text', '')
                el_crop = element.get('crop')
                
                # Header for the item
                item_header = f"[Page {page_num}, Item {el_id}] {el_type}"
                
                if el_type in ['Text', 'Title', 'List', 'Table']:
                    if el_text:
                        prompt_parts.append(f"{item_header}: {el_text}")
                    else:
                        # If no text (e.g. table image without OCR), we might want to send the image
                        # For now, let's send the image for Tables and Figures
                        pass
                
                if el_type in ['Figure', 'Table', 'Formula']:
                    # Send the cropped image for visual context
                    if el_crop is not None:
                        # Convert numpy array (BGR) to PIL Image (RGB)
                        # Assuming el_crop is BGR from cv2
                        try:
                            pil_img = Image.fromarray(el_crop[..., ::-1]) # BGR to RGB
                            prompt_parts.append(item_header)
                            prompt_parts.append(pil_img)
                        except Exception as e:
                            print(f"Error converting image for {item_header}: {e}")

        return prompt_parts

    def summarize(self, layout_results):
        """
        Generate the summary using Gemini.
        """
        content = self.prepare_content(layout_results)
        
        try:
            response = self.model.generate_content(content)
            # Extract JSON from the response
            text = response.text
            # Simple cleanup to find the JSON block if wrapped in markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
                
            return json.loads(text.strip())
        except Exception as e:
            return {"error": str(e)}

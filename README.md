# 📝 Research Vision: Intelligent PDF Analysis & Summarization Pipeline

**Course:** Image and Video Processing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-8E75B2)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8)
![LayoutParser](https://img.shields.io/badge/Layout-LayoutParser-00C853)

## 📖 Introduction

**Research Vision** is a state-of-the-art document processing system developed for the **Image and Video Processing** course. It addresses the challenge of extracting structured knowledge from complex academic papers, which often contain a mix of dense text, multi-column layouts, mathematical formulas, and visual data (figures, tables).

Traditional PDF parsers often fail to preserve the semantic structure of such documents. This project solves this by implementing a robust pipeline that combines:
1.  **Classical Computer Vision** for image enhancement and noise reduction.
2.  **Deep Learning (Detectron2)** for semantic layout analysis and object detection.
3.  **Generative AI (Gemini 1.5 Flash)** for context-aware summarization and reasoning.

The result is an interactive tool that transforms static PDFs into intelligent, queryable, and summarized insights.

---

## 🚀 Key Features & Technical Details

### 1. �️ Advanced Image Preprocessing
Before analysis, every PDF page is converted to a high-resolution image and undergoes a rigorous enhancement pipeline to ensure optimal OCR and detection performance.
*   **Grayscale Conversion**: Reduces computational complexity by converting 3-channel BGR images to single-channel grayscale.
*   **Contrast Enhancement**:
    *   **Histogram Equalization**: Global contrast improvement.
    *   **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Local contrast enhancement to bring out details in dark or over-exposed regions without amplifying noise.
*   **Noise Reduction**:
    *   **Gaussian Blur**: Smooths out high-frequency noise.
    *   **Median Blur**: Effectively removes salt-and-pepper noise while preserving edge details.
*   **Binarization**:
    *   **Otsu's Thresholding**: Automatically calculates the optimal global threshold to separate foreground (text) from background.
    *   **Adaptive Thresholding**: Calculates thresholds for small regions, handling varying illumination conditions across the page.

### 2. 🧩 Deep Layout Analysis
We utilize **LayoutParser** powered by a **Detectron2** backend to perform Object Detection on document images.
*   **Model Architecture**: Mask R-CNN (ResNet-50-FPN backbone) trained on the **PubLayNet** dataset.
*   **Detected Classes**:
    *   `Text`: Paragraphs and body text.
    *   `Title`: Section headers and paper titles.
    *   `List`: Bullet points and enumerated lists.
    *   `Table`: Tabular data regions.
    *   `Figure`: Charts, graphs, and diagrams.
*   **Configuration**:
    *   **Config**: `config.yaml` (Customized Detectron2 configuration).
    *   **Weights**: `model_final.pth` (Pre-trained weights loaded locally).

### 3. 🔍 Optical Character Recognition (OCR)
*   **Engine**: **Tesseract OCR** (via `pytesseract`).
*   **Function**: Extracts raw text content from the bounding boxes identified by the layout model (specifically for `Text`, `Title`, and `List` classes).

### 4. 🤖 AI-Powered Summarization
*   **Model**: **Google Gemini 1.5 Flash**.
*   **Multimodal Capability**: The model receives both the extracted text and the cropped images of figures/tables/formulas.
*   **Prompt Engineering**:
    *   **Per-Page Analysis**: Generates a summary for each individual page.
    *   **Contextual Citation**: The model is instructed to cite sources using specific IDs (e.g., `(Source: Page 1, Item 5)`), linking insights back to their original location in the document.
    *   **Structured Output**: Returns a JSON object containing both the overall summary and detailed page-wise key points.

### 5. � Interactive Dashboard
*   **Framework**: **Streamlit**.
*   **Functionality**:
    *   **Upload**: Drag-and-drop PDF interface.
    *   **Preprocessing Lab**: Visual comparison of original vs. processed images (Denoised, Thresholded, etc.).
    *   **Layout Inspector**: Interactive visualization of bounding boxes and class labels.
    *   **Summary View**: Clean, readable display of the AI-generated insights.

---

## 📂 Project Structure

```
research-vision/
├── app/
│   └── streamlit_preprocess.py  # 🚀 Main Application: Streamlit UI & Logic
├── core/
│   ├── gemini/
│   │   └── summarizer.py        # � AI Module: Gemini API integration & Prompting
│   ├── layout/
│   │   └── layout_analysis.py   # 👁️ Vision Module: LayoutParser & Detectron2 wrapper
│   ├── preprocessing/
│   │   └── preprocess.py        # ⚡ CV Module: OpenCV image enhancement algorithms
│   └── pdf/
│       └── pdf_loader.py        # 📄 Utility: PDF to Image conversion
├── data/                        # � Input: Temporary storage for uploaded PDFs
├── notebooks/
│   └── models/                  # 📦 Models: Local storage for model weights & config
│       ├── config.yaml
│       └── model_final.pth
├── outputs/                     # � Output: Generated crops, images, and JSONs
├── requirements.txt             # �️ Dependencies: Python package list
└── README.md                    # 📖 Documentation: This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
1.  **Python 3.8+** installed.
2.  **Tesseract OCR** installed and added to your system PATH.
    *   [Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)
    *   Linux: `sudo apt-get install tesseract-ocr`
    *   macOS: `brew install tesseract`

### Step 1: Clone the Repository
```bash
git clone https://github.com/anushapundir/research-vision.git
cd research-vision
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you face issues installing `detectron2` or `layoutparser`, refer to their official installation guides as they may require specific PyTorch versions.*

### Step 4: Model Setup ⚠️ **IMPORTANT**
You must manually place the model weights and configuration file in the `notebooks/models/` directory.
1.  **Download** the PubLayNet model weights (`model_final.pth`) and config (`config.yaml`).
2.  **Place them** in:
    ```
    research-vision/
    └── notebooks/
        └── models/
            ├── config.yaml
            └── model_final.pth
    ```

### Step 5: Run the Application
```bash
streamlit run app/streamlit_preprocess.py
```

---

## 🎮 Usage Guide

1.  **Start the App**: Run the command above. The interface will open in your browser at `http://localhost:8501`.
2.  **Upload PDF**: Use the sidebar file uploader to select a research paper.
3.  **Preprocessing Tab**:
    *   Select a page to view.
    *   Toggle between different views: "Original", "Grayscale", "Equalized", "Denoised", "Binary (Otsu)", "Binary (Adaptive)".
    *   *Goal: Verify that the image quality is sufficient for OCR.*
4.  **Layout Analysis Tab**:
    *   Click **"Run Layout Analysis"** in the sidebar.
    *   Wait for the model to process the pages.
    *   View the annotated images with bounding boxes.
    *   Explore the "Extracted Elements" section to see individual crops of figures and tables.
5.  **Summarization Tab**:
    *   Enter your **Google Gemini API Key** in the sidebar.
    *   Click **"Generate Summary"**.
    *   Read the "Overall Summary" and "Page-wise Breakdown".

---

## � Troubleshooting

*   **Tesseract Not Found**:
    *   Error: `TesseractNotFoundError: tesseract is not installed or it's not in your PATH`
    *   **Fix**: Install Tesseract and ensure the executable path is added to your system's Environment Variables. You may need to restart your terminal/IDE.
*   **Model Config Error**:
    *   Error: `AssertionError: Config file ... does not exist!`
    *   **Fix**: Double-check that `config.yaml` and `model_final.pth` are exactly in `notebooks/models/`.
*   **Gemini API Error**:
    *   Error: `404 models/gemini-2.5-flash is not found`
    *   **Fix**: Ensure your API key is valid and has access to the `gemini-2.5-flash` model. You can generate a key at [Google AI Studio](https://aistudio.google.com/).

---

## 💻 Technical Stack

*   **Language**: Python 3.9
*   **Frontend**: Streamlit
*   **Computer Vision**: OpenCV (`cv2`), NumPy
*   **Deep Learning**: PyTorch, Detectron2, LayoutParser
*   **OCR**: Tesseract (`pytesseract`)
*   **GenAI**: Google Generative AI SDK (`google-generativeai`)
*   **PDF Handling**: PyMuPDF (`fitz`), pdf2image

---

*Developed by Anusha Pundir and Paarangat Rai Sharma for the Image and Video Processing Course Project.*

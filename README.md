# Research Vision

**PDF document analysis and summarization using classical Image Processing, OCR, and AI.**

An academic project for Image and Video Processing course that implements an end-to-end pipeline for research paper analysis.

## 🚀 Quick Start

### First-Time Setup

If you cloned or have the old structure, clean up legacy files:
```powershell
.\cleanup.ps1
```

### Installation

```powershell
pip install -r requirements.txt
```

### Run Preprocessing Web App

```powershell
streamlit run app/streamlit_preprocess.py
```

Access the app at `http://localhost:8501`

### Run CLI Demo

```powershell
python scripts/demo.py
```

## 📋 Features

### ✅ Implemented (Phase 1: Preprocessing)

- **PDF to Image Conversion** - High-quality rendering at configurable DPI using PyMuPDF
- **Classical Image Processing Pipeline**:
  - Grayscale conversion
  - Histogram equalization (global & CLAHE)
  - Gaussian blur denoising
  - Median blur denoising
  - Otsu's thresholding
  - Adaptive thresholding
- **Interactive Web Interface** - View and compare all preprocessing stages
- **Batch Processing** - All pages processed automatically
- **Export Capabilities** - Download processed images

### 🚧 Planned (Future Phases)

- **Layout Segmentation** - Text block and figure detection using morphological operations
- **OCR Integration** - Text extraction with Tesseract/EasyOCR
- **AI Summarization** - Research paper summarization using Google Gemini
- **End-to-End Pipeline** - Complete document analysis workflow

## 🏗️ Project Structure

```
research-vision/
│
├── app/
│   ├── streamlit_preprocess.py     # Preprocessing web interface (✅ Working)
│   └── streamlit_full_app.py       # Full pipeline interface (⏳ Placeholder)
│
├── core/
│   ├── pdf/
│   │   └── pdf_loader.py           # PDF to image conversion (✅ Implemented)
│   │
│   ├── preprocessing/
│   │   └── preprocess.py           # Image processing filters (✅ Implemented)
│   │
│   ├── segmentation/
│   │   ├── segmenter.py            # Layout analysis (⏳ Placeholder)
│   │   └── morphology_utils.py     # Morphological operations (⏳ Placeholder)
│   │
│   ├── ocr/
│   │   ├── ocr_engine.py           # Text extraction (⏳ Placeholder)
│   │   └── postprocess_text.py     # Text cleaning (⏳ Placeholder)
│   │
│   ├── ai/
│   │   ├── gemini_client.py        # Gemini API interface (⏳ Placeholder)
│   │   └── summary_pipeline.py     # Summarization pipeline (⏳ Placeholder)
│   │
│   └── pipeline/
│       └── end_to_end.py           # Full pipeline orchestration (⏳ Placeholder)
│
├── data/                           # Input PDFs
├── outputs/                        # Processing results
│   ├── pages/                      # Original page images
│   ├── preprocessed/               # Filtered images
│   ├── segments/                   # Segmented regions
│   ├── crops_text/                 # Text block crops
│   ├── crops_figures/              # Figure crops
│   ├── ocr_text/                   # Extracted text
│   └── summaries/                  # AI-generated summaries
│
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── preprocessing_tests.ipynb
│   ├── segmentation_debug.ipynb
│   └── ocr_quality_tests.ipynb
│
├── scripts/                        # Utility scripts
│   └── demo.py                     # CLI demo script
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔬 Preprocessing Pipeline (Current Implementation)

```
PDF → Load Pages → Grayscale → Histogram Equalization → Denoising → Binarization → Output
                      ↓              ↓                      ↓           ↓
                  (weighted)    (global & CLAHE)    (Gaussian/Median) (Otsu/Adaptive)
```

### Image Processing Techniques

1. **Grayscale Conversion** - Weighted average: `0.299*R + 0.587*G + 0.114*B`
2. **Histogram Equalization** - Global contrast enhancement
3. **CLAHE** - Contrast Limited Adaptive Histogram Equalization
4. **Gaussian Blur** - Noise reduction with Gaussian kernel
5. **Median Blur** - Salt-and-pepper noise removal
6. **Otsu's Thresholding** - Automatic global binarization
7. **Adaptive Thresholding** - Local binarization for varying illumination

## 📊 Usage Examples

### Preprocessing a PDF (CLI)

```python
from core.pdf.pdf_loader import load_pdf_to_images
from core.preprocessing.preprocess import run_full_preprocessing

# Load PDF
pages = load_pdf_to_images("paper.pdf", dpi=200)

# Preprocess first page
results = run_full_preprocessing(pages[0])

# Access different outputs
grayscale = results['gray']
binary = results['otsu']
adaptive = results['adaptive']
```

### Using Individual Filters

```python
from core.preprocessing.preprocess import (
    to_grayscale,
    equalize_clahe,
    binarize_otsu
)

gray = to_grayscale(page_image)
enhanced = equalize_clahe(gray, clip_limit=2.0, tile_size=8)
binary = binarize_otsu(enhanced)
```

## 🛠️ Tech Stack

### Current
- **Python 3.8+** - Core language
- **OpenCV** - Image processing operations
- **PyMuPDF (fitz)** - High-quality PDF rendering
- **NumPy** - Numerical array operations
- **Streamlit** - Interactive web interface
- **Pillow** - Additional image handling

### Planned
- **Tesseract/EasyOCR** - Optical character recognition
- **Google Gemini API** - AI-powered summarization
- **Matplotlib** - Visualization in notebooks

## 🧪 Development

### Jupyter Notebooks

Use the provided notebooks for experimentation:

```powershell
# Install Jupyter if needed
pip install jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

- **preprocessing_tests.ipynb** - Test different preprocessing parameters
- **segmentation_debug.ipynb** - Develop segmentation algorithms
- **ocr_quality_tests.ipynb** - Compare OCR engines and quality

### Project Status

**Phase 1: Preprocessing** ✅ Complete
- All classical image processing filters implemented
- Web and CLI interfaces working
- Batch processing and export functional

**Phase 2: Segmentation** ⏳ In Progress
- Morphological operations to be implemented
- Connected component analysis
- Text vs. figure classification

**Phase 3: OCR** ⏳ Planned
- OCR engine integration
- Text cleaning and postprocessing
- Multi-block text merging

**Phase 4: AI Summarization** ⏳ Planned
- Gemini API integration
- Prompt engineering for research papers
- Structured summary generation

**Phase 5: Full Pipeline** ⏳ Planned
- End-to-end orchestration
- Complete web interface
- Batch document processing

## 📝 Notes

- This is an academic project for learning classical image processing techniques
- All preprocessing logic is preserved from the original implementation
- The modular structure allows for easy experimentation and extension
- Placeholder modules have clear documentation and type hints for future implementation

## 👥 Contributing

This is an educational project. When implementing future modules:

1. Follow the existing code style and structure
2. Add comprehensive docstrings with type hints
3. Include usage examples in docstrings
4. Test in the corresponding Jupyter notebook
5. Update this README with implementation details

## 📄 License

Educational project for Image and Video Processing course.

---

**Version**: 2.0.0  
**Research Vision** - From Document to Insight

**Status**: Phase 1 Complete | Phases 2-5 In Development

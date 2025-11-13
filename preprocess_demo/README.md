# Research Vision

PDF document preprocessing using classical Image and Video Processing techniques.

## 🚀 Quick Start

### Installation

```powershell
pip install -r requirements.txt
```

### Run Web Application

```powershell
python -m streamlit run app_streamlit_preprocess.py
```

Access the app at `http://localhost:8501`

### Run CLI Demo

```powershell
python main_demo.py
```

## 📋 Features

- **PDF to Image Conversion** - High-quality rendering at configurable DPI
- **Automatic Processing** - All pages preprocessed on upload
- **Classical Image Processing Pipeline**:
  - Grayscale conversion
  - Histogram equalization (global & CLAHE)
  - Gaussian blur denoising
  - Median blur denoising
  - Otsu's thresholding
  - Adaptive thresholding
- **Interactive Web Interface** - View and compare all preprocessing stages
- **Batch Export** - Download processed images

## 🏗️ Project Structure

```
preprocess_demo/
├── app/
│   ├── pdf_loader.py       # PDF to image conversion
│   └── preprocess.py       # Image processing pipeline
├── data/                   # Upload PDFs here
├── outputs/preprocess/     # CLI output directory
├── main_demo.py           # CLI interface
└── app_streamlit_preprocess.py  # Web interface
```

## 🔬 Preprocessing Pipeline

```
PDF → Load Pages → Grayscale → Histogram Equalization → Denoising → Binarization → Output
```

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV - Image processing
- PyMuPDF - PDF rendering
- Streamlit - Web interface
- NumPy - Array operations

## � License

Educational project for Image and Video Processing.

---

**Version**: 1.0.0  
**Research Vision** - Enhanced Document Processing

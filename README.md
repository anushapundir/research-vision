# 📝 Research Vision: Intelligent PDF Analysis & Summarization Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Gemini](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-8E75B2)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8)
![LayoutParser](https://img.shields.io/badge/Layout-LayoutParser-00C853)

## 📖 Introduction

**Research Vision** is an advanced document processing system designed to transform static PDF research papers into interactive, structured, and summarized insights. In the academic and research domain, manually extracting information from complex layouts containing text, figures, tables, and formulas is a time-consuming process. 

This project automates this workflow by combining **Classical Computer Vision** techniques for image preprocessing with **Deep Learning** models for layout analysis and **Generative AI** for semantic understanding. The result is a streamlined pipeline that not only enhances document readability but also provides intelligent, context-aware summaries.

---

## 🚀 Key Features

*   **📄 Automated PDF Preprocessing**: Converts PDF pages into high-quality images and applies a suite of image enhancement techniques (Denoising, Thresholding, Contrast Enhancement) to improve OCR accuracy.
*   **🧩 Deep Layout Analysis**: Utilizes **LayoutParser** with a **Detectron2** backend to intelligently segment pages into semantic regions:
    *   Text Blocks
    *   Titles & Headers
    *   Lists
    *   Tables
    *   Figures & Diagrams
*   **🔍 Optical Character Recognition (OCR)**: Integrates **Tesseract OCR** to extract raw text from identified text regions with high precision.
*   **🤖 AI-Powered Summarization**: Leverages **Google's Gemini 1.5 Flash** model to generate comprehensive summaries. It analyzes both the extracted text and the visual content (figures/tables) to provide a holistic understanding of the paper.
*   **📊 Interactive Dashboard**: A user-friendly **Streamlit** interface that allows users to visualize every step of the pipeline, from raw image processing to final AI-generated insights.

---

## 🛠️ System Architecture & Pipeline

The project follows a modular pipeline approach, ensuring each stage transforms the data for the next.

```mermaid
graph TD
    subgraph Input
    A[User Uploads PDF]
    end

    subgraph "Stage 1: Preprocessing"
    A --> B[PDF Loader & Rasterization]
    B --> C[Grayscale Conversion]
    C --> D[Contrast Enhancement <br/> (Histogram Eq / CLAHE)]
    D --> E[Noise Reduction <br/> (Gaussian & Median Blur)]
    E --> F[Binarization <br/> (Otsu & Adaptive Thresholding)]
    end

    subgraph "Stage 2: Layout Analysis"
    F --> G[LayoutParser Model <br/> (Detectron2)]
    G --> H[Element Detection]
    H --> I{Element Classification}
    I -->|Text/Title/List| J[Tesseract OCR]
    I -->|Figure/Table| K[Image Cropping]
    end

    subgraph "Stage 3: AI Synthesis"
    J & K --> L[Context Aggregation]
    L --> M[Gemini 1.5 Flash API]
    M --> N[Structured Summary <br/> (JSON)]
    end

    subgraph Output
    N --> O[Streamlit Dashboard]
    O --> P[Visualizations & Insights]
    end
```

---

## 💻 Technical Stack

### Core Frameworks
*   **Python**: Primary programming language.
*   **Streamlit**: For building the interactive web application.

### Computer Vision & Image Processing
*   **OpenCV (cv2)**: Used for all low-level image manipulation (thresholding, blurring, etc.).
*   **LayoutParser**: A unified toolkit for Deep Learning-based document image analysis.
*   **Detectron2**: The underlying object detection library used by LayoutParser.
*   **Tesseract OCR**: For extracting text from image regions.

### Artificial Intelligence
*   **Google Gemini API**: Uses the `gemini-1.5-flash` model for multimodal (text + image) reasoning and summarization.

---

## 📂 Project Structure

The codebase is organized into modular components for maintainability and scalability.

```
research-vision/
├── app/
│   └── streamlit_preprocess.py  # 🚀 Main Streamlit Application entry point
├── core/
│   ├── gemini/
│   │   └── summarizer.py        # 🤖 Gemini API integration & prompt engineering
│   ├── layout/
│   │   └── layout_analysis.py   # 🧩 LayoutParser model wrapper & element extraction
│   ├── preprocessing/
│   │   └── preprocess.py        # 🖼️ Classical CV algorithms (Blur, Threshold, etc.)
│   └── pdf/
│       └── pdf_loader.py        # 📄 PDF to Image conversion utilities
├── data/                        # 📁 Temporary storage for uploaded PDFs
├── notebooks/
│   └── models/                  # 🧠 Pre-trained Detectron2 model weights & config
├── outputs/                     # 💾 Generated results (crops, visualizations, JSONs)
├── requirements.txt             # 📦 Project dependencies
└── README.md                    # 📖 Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to set up the project locally.

### Prerequisites
*   Python 3.8 or higher
*   Tesseract OCR installed on your system ([Installation Guide](https://github.com/tesseract-ocr/tesseract))

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/research-vision.git
cd research-vision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Setup
Ensure the LayoutParser model weights (`model_final.pth`) and configuration (`config.yaml`) are placed in `notebooks/models/`.

### 4. Run the Application
```bash
streamlit run app/streamlit_preprocess.py
```

---

## 🎮 Usage Guide

1.  **Upload**: Launch the app and upload a research paper (PDF) via the sidebar.
2.  **Preprocessing View**: Explore the "Preprocessing" tab to see how computer vision techniques enhance the document image. You can download specific outputs like the binarized or denoised versions.
3.  **Layout Analysis**: Click **"Run Layout Analysis"** in the sidebar. The system will detect and bound regions of interest. View these in the "Layout Analysis" tab, where you can inspect individual cropped elements (figures, tables, text).
4.  **Generate Summary**: Enter your **Gemini API Key** in the sidebar and click **"Generate Summary"**. The AI will read the extracted text and look at the figures to produce a detailed, page-by-page and overall summary of the paper.

---

## 🔮 Future Scope

*   **Fine-tuned Models**: Training the layout analysis model specifically on scientific papers for higher accuracy.
*   **Knowledge Graph Generation**: Mapping relationships between entities extracted from the text.
*   **Chat with PDF**: Implementing a RAG (Retrieval-Augmented Generation) system to allow users to ask specific questions about the paper.
*   **Multi-Document Support**: Analyzing and cross-referencing multiple papers simultaneously.

---

*Developed for [Course Name/College Name] Project.*

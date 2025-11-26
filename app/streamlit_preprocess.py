"""
Research Vision - PDF Preprocessing Application
Interactive web app to visualize PDF page preprocessing results.

Usage:
    streamlit run app/streamlit_preprocess.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Configuration - Use paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Updated imports for new structure
from core.pdf.pdf_loader import load_pdf_to_images
from core.preprocessing.preprocess import run_full_preprocessing
from core.layout.layout_analysis import LayoutAnalyzer
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PAGES_DIR = OUTPUTS_DIR / "pages"
PREPROCESSED_DIR = OUTPUTS_DIR / "preprocessed"
LAYOUT_ANALYSIS_DIR = OUTPUTS_DIR / "layout_analysis"

# Model Paths
MODEL_CONFIG_PATH = PROJECT_ROOT / "notebooks/models/config.yaml"
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "notebooks/models/model_final.pth"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LAYOUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_layout_analyzer():
    """
    Initialize and cache the LayoutAnalyzer.
    """
    if not MODEL_CONFIG_PATH.exists() or not MODEL_WEIGHTS_PATH.exists():
        st.error("❌ Model files not found! Please check notebooks/models/")
        return None
        
    return LayoutAnalyzer(
        config_path=str(MODEL_CONFIG_PATH),
        model_path=str(MODEL_WEIGHTS_PATH)
    )


def numpy_to_streamlit(image: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR/Grayscale image to RGB for Streamlit display.
    
    Args:
        image (np.ndarray): OpenCV image (BGR or grayscale)
        
    Returns:
        np.ndarray: RGB image for Streamlit
    """
    if len(image.shape) == 2:
        # Grayscale - convert to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_preprocessing_outputs(page_idx: int, results: dict) -> None:
    """
    Save preprocessing outputs to the outputs directory structure.
    
    Args:
        page_idx (int): Page index (0-based)
        results (dict): Preprocessing results dictionary
    """
    # Save original page
    page_path = PAGES_DIR / f"page_{page_idx + 1:03d}.png"
    cv2.imwrite(str(page_path), results['original'])
    
    # Save key preprocessed outputs
    for name in ['gray', 'otsu', 'adaptive']:
        if name in results:
            output_path = PREPROCESSED_DIR / f"page_{page_idx + 1:03d}_{name}.png"
            cv2.imwrite(str(output_path), results[name])


# Import Gemini Summarizer
from core.gemini.summarizer import GeminiSummarizer

def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Research Vision - PDF Preprocessor",
        page_icon="📄",
        layout="wide"
    )
    
    # Title and description
    st.title("📝 Research Vision")
    st.markdown("### PDF Document Preprocessing Pipeline")
    st.markdown(
        "Upload a PDF document and apply classical image processing techniques for enhanced text extraction."
    )
    
    st.divider()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF File",
            type=["pdf"],
            help="Select a PDF file to process"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            pdf_path = DATA_DIR / uploaded_file.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            
            # DPI selector
            dpi = st.slider(
                "Resolution (DPI)",
                min_value=100,
                max_value=300,
                value=200,
                step=50,
                help="Higher DPI = better quality but slower processing"
            )
            
            # Load PDF only if not already loaded or if DPI changed
            should_load = (
                'pages' not in st.session_state or 
                st.session_state.get('current_pdf') != uploaded_file.name or
                st.session_state.get('current_dpi') != dpi
            )
            
            if should_load:
                with st.spinner("Loading PDF pages..."):
                    try:
                        pages = load_pdf_to_images(str(pdf_path), dpi=dpi)
                        st.session_state['pages'] = pages
                        st.session_state['current_pdf'] = uploaded_file.name
                        st.session_state['current_dpi'] = dpi
                        st.session_state['pdf_loaded'] = True
                        # Process all pages immediately
                        st.session_state['all_results'] = {}
                        for idx, page in enumerate(pages):
                            results = run_full_preprocessing(page)
                            st.session_state['all_results'][idx] = results
                            # Save outputs to disk
                            save_preprocessing_outputs(idx, results)
                    except Exception as e:
                        st.error(f"❌ Error loading PDF: {str(e)}")
                        st.session_state['pdf_loaded'] = False
                        return
            
            pages = st.session_state.get('pages', [])
            st.success(f"✓ Loaded and processed {len(pages)} page(s)")
            st.info(f"📁 Outputs saved to:\n- `{PAGES_DIR.relative_to(PROJECT_ROOT)}/`\n- `{PREPROCESSED_DIR.relative_to(PROJECT_ROOT)}/`")
            
            # Page selector
            if st.session_state.get('pdf_loaded', False) and len(pages) > 0:
                page_index = st.selectbox(
                    "Select Page to View",
                    options=range(len(pages)),
                    format_func=lambda x: f"Page {x + 1}",
                    help="Choose which page to view"
                )
                st.session_state['page_index'] = page_index
        else:
            st.info("👆 Upload a PDF file to get started")
            
        st.divider()
        st.header("🧩 Layout Analysis")
        enable_layout_analysis = st.checkbox("Enable Layout Analysis", help="Detect text, tables, figures, etc.")
        
        if enable_layout_analysis and st.session_state.get('pdf_loaded', False):
            if st.button("Run Layout Analysis"):
                analyzer = get_layout_analyzer()
                if analyzer:
                    with st.spinner("Running Layout Analysis on all pages..."):
                        layout_results = {}
                        pages = st.session_state.get('pages', [])
                        pdf_name = st.session_state.get('current_pdf', 'document')
                        pdf_stem = Path(pdf_name).stem
                        
                        # Create output directory for this PDF
                        pdf_layout_dir = LAYOUT_ANALYSIS_DIR / pdf_stem
                        pdf_layout_dir.mkdir(parents=True, exist_ok=True)
                        
                        for idx, page in enumerate(pages):
                            # Create page directory
                            page_dir = pdf_layout_dir / f"page_{idx + 1:03d}"
                            page_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Run detection
                            layout = analyzer.process_image(page)
                            
                            # Extract elements
                            result = analyzer.extract_elements(
                                page, 
                                layout, 
                                output_dir=page_dir,
                                page_name=f"page_{idx + 1:03d}"
                            )
                            
                            # Save visualization
                            viz_path = page_dir / "visualization.png"
                            cv2.imwrite(str(viz_path), result["visualization"])
                            
                            layout_results[idx] = result
                            
                        st.session_state['layout_results'] = layout_results
                        st.success("✓ Layout Analysis Complete!")
                        st.info(f"📁 Results saved to: `{pdf_layout_dir.relative_to(PROJECT_ROOT)}/`")

        st.divider()
        st.header("🤖 Gemini Summarization")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
        
        if gemini_api_key and st.session_state.get('layout_results'):
            if st.button("Generate Summary"):
                with st.spinner("Generating summary with Gemini..."):
                    summarizer = GeminiSummarizer(gemini_api_key)
                    summary_result = summarizer.summarize(st.session_state['layout_results'])
                    
                    if "error" in summary_result:
                        st.error(f"Error generating summary: {summary_result['error']}")
                    else:
                        st.session_state['summary_result'] = summary_result
                        st.success("Summary Generated!")
    
    # Main content area
    if not uploaded_file:
        st.info("👈 Please upload a PDF file from the sidebar to begin.")
        
        # Show usage instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Upload PDF**: Click "Browse files" in the sidebar
            2. **Auto-Processing**: All pages are automatically preprocessed
            3. **Select Page**: Choose which page to view from the dropdown
            4. **Download**: Export processed images for any page
            5. **Outputs**: All results are automatically saved to the `outputs/` directory
            
            **Preprocessing Pipeline:**
            - Grayscale conversion
            - Histogram equalization (global and CLAHE)
            - Gaussian blur (noise reduction)
            - Median blur (salt-and-pepper noise removal)
            - Otsu's thresholding (automatic binary conversion)
            - Adaptive thresholding (local binary conversion)
            """)
        
        return
    
    # Display the selected page results
    if st.session_state.get('pdf_loaded', False):
        page_index = st.session_state.get('page_index', 0)
        all_results = st.session_state.get('all_results', {})
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Preprocessing", "Layout Analysis", "Summaries"])
        
        with tab1:
            if page_index in all_results:
                results = all_results[page_index]
                
                st.subheader(f"📄 Page {page_index + 1} Results")
                
                # Display results in a grid layout
                st.divider()
                st.subheader("📊 Preprocessing Results")
                
                # Row 1: Original and Grayscale
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Image**")
                    st.image(numpy_to_streamlit(results['original']), use_column_width=True)
                with col2:
                    st.markdown("**Grayscale**")
                    st.caption("Converted using weighted average: 0.299R + 0.587G + 0.114B")
                    st.image(numpy_to_streamlit(results['gray']), use_column_width=True)
                
                # Row 2: Histogram Equalization
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Histogram Equalized**")
                    st.caption("Global contrast enhancement using histogram redistribution")
                    st.image(numpy_to_streamlit(results['equalized']), use_column_width=True)
                with col2:
                    st.markdown("**CLAHE Enhanced**")
                    st.caption("Contrast Limited Adaptive Histogram Equalization")
                    st.image(numpy_to_streamlit(results['clahe']), use_column_width=True)
                
                # Row 3: Denoising
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Gaussian Blur**")
                    st.caption("Weighted average smoothing for Gaussian noise reduction")
                    st.image(numpy_to_streamlit(results['gaussian']), use_column_width=True)
                with col2:
                    st.markdown("**Median Blur**")
                    st.caption("Non-linear filter excellent for salt-and-pepper noise")
                    st.image(numpy_to_streamlit(results['median']), use_column_width=True)
                
                # Row 4: Binarization
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Otsu's Thresholding**")
                    st.caption("Automatic global threshold using bimodal histogram")
                    st.image(numpy_to_streamlit(results['otsu']), use_column_width=True)
                with col2:
                    st.markdown("**Adaptive Thresholding**")
                    st.caption("Local thresholding for varying illumination")
                    st.image(numpy_to_streamlit(results['adaptive']), use_column_width=True)
                
                # Download section
                st.divider()
                st.subheader("💾 Download Processed Images")
                
                col1, col2, col3 = st.columns(3)
                
                # Prepare download buttons for key outputs
                with col1:
                    _, buffer = cv2.imencode('.png', results['gray'])
                    st.download_button(
                        label="⬇️ Download Grayscale",
                        data=buffer.tobytes(),
                        file_name=f"page_{page_index + 1}_grayscale.png",
                        mime="image/png"
                    )
                
                with col2:
                    _, buffer = cv2.imencode('.png', results['otsu'])
                    st.download_button(
                        label="⬇️ Download Otsu Binary",
                        data=buffer.tobytes(),
                        file_name=f"page_{page_index + 1}_otsu.png",
                        mime="image/png"
                    )
                
                    st.download_button(
                        label="⬇️ Download Adaptive Binary",
                        data=buffer.tobytes(),
                        file_name=f"page_{page_index + 1}_adaptive.png",
                        mime="image/png"
                    )
        
        with tab2:
            # Layout Analysis Results
            layout_results = st.session_state.get('layout_results', {})
            if page_index in layout_results:
                st.subheader("🧩 Layout Analysis Results")
                
                l_result = layout_results[page_index]
                
                # Visualization
                st.markdown("**Detection Visualization**")
                st.image(numpy_to_streamlit(l_result['visualization']), use_column_width=True)
                
                # Detected Elements
                elements = l_result['elements']
                if elements:
                    st.markdown(f"**Detected {len(elements)} Elements**")
                    
                    # Group by type
                    grouped_elements = {}
                    for el in elements:
                        etype = el['type']
                        if etype not in grouped_elements:
                            grouped_elements[etype] = []
                        grouped_elements[etype].append(el)
                    
                    # Create tabs for each type
                    type_tabs = st.tabs(list(grouped_elements.keys()))
                    
                    for i, (etype, items) in enumerate(grouped_elements.items()):
                        with type_tabs[i]:
                            for item in items:
                                with st.expander(f"{etype} #{item['id']}"):
                                    c1, c2 = st.columns([1, 2])
                                    with c1:
                                        st.image(numpy_to_streamlit(item['crop']), use_column_width=True)
                                    with c2:
                                        if item['text']:
                                            st.markdown("**Extracted Text:**")
                                            st.text_area("OCR Output", item['text'], height=150, key=f"text_{page_index}_{item['id']}")
                                        else:
                                            st.info("No text extracted or not a text region.")
            else:
                st.info("Run Layout Analysis to see results here.")

        with tab3:
            summary_result = st.session_state.get('summary_result')
            if summary_result:
                st.subheader("📝 Research Paper Summary")
                
                st.markdown("### Overall Summary")
                st.write(summary_result.get('overall_summary', 'No summary available.'))
                
                st.divider()
                st.markdown("### Per-Page Summaries")
                
                page_summaries = summary_result.get('page_summaries', [])
                for p_summary in page_summaries:
                    p_num = p_summary.get('page_number')
                    st.markdown(f"#### Page {p_num}")
                    st.write(p_summary.get('summary', ''))
                    
                    st.markdown("**Key Points:**")
                    for point in p_summary.get('key_points', []):
                        st.markdown(f"- {point}")
                        
                        # Try to extract source reference
                        # Format: (Source: Page X, Item Y)
                        import re
                        match = re.search(r"Source: Page (\d+), Item (\d+)", point)
                        if match:
                            src_page = int(match.group(1))
                            src_item = int(match.group(2))
                            
                            # Find the element
                            if (src_page - 1) in layout_results:
                                src_elements = layout_results[src_page - 1]['elements']
                                for el in src_elements:
                                    if el['id'] == src_item:
                                        with st.expander(f"View Source: {el['type']} #{el['id']}"):
                                            st.image(numpy_to_streamlit(el['crop']), caption=f"Source for: {point[:50]}...")
            else:
                st.info("Enter Gemini API Key and click 'Generate Summary' to see results here.")

if __name__ == "__main__":
    main()

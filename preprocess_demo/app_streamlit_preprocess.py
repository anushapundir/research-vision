"""
Research Vision - PDF Preprocessing Application
Interactive web app to visualize PDF page preprocessing results.

Usage:
    python -m streamlit run app_streamlit_preprocess.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from app.pdf_loader import load_pdf_to_images
from app.preprocess import run_full_preprocessing


# Configuration
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


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


def main():
    """
    Main Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Research Vision - PDF Preprocessor",
        page_icon="�",
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
            pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
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
                        pages = load_pdf_to_images(pdf_path, dpi=dpi)
                        st.session_state['pages'] = pages
                        st.session_state['current_pdf'] = uploaded_file.name
                        st.session_state['current_dpi'] = dpi
                        st.session_state['pdf_loaded'] = True
                        # Process all pages immediately
                        st.session_state['all_results'] = {}
                        for idx, page in enumerate(pages):
                            st.session_state['all_results'][idx] = run_full_preprocessing(page)
                    except Exception as e:
                        st.error(f"❌ Error loading PDF: {str(e)}")
                        st.session_state['pdf_loaded'] = False
                        return
            
            pages = st.session_state.get('pages', [])
            st.success(f"✓ Loaded and processed {len(pages)} page(s)")
            
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
                st.image(numpy_to_streamlit(results['original']), use_container_width=True)
            with col2:
                st.markdown("**Grayscale**")
                st.caption("Converted using weighted average: 0.299R + 0.587G + 0.114B")
                st.image(numpy_to_streamlit(results['gray']), use_container_width=True)
            
            # Row 2: Histogram Equalization
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Histogram Equalized**")
                st.caption("Global contrast enhancement using histogram redistribution")
                st.image(numpy_to_streamlit(results['equalized']), use_container_width=True)
            with col2:
                st.markdown("**CLAHE Enhanced**")
                st.caption("Contrast Limited Adaptive Histogram Equalization")
                st.image(numpy_to_streamlit(results['clahe']), use_container_width=True)
            
            # Row 3: Denoising
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Gaussian Blur**")
                st.caption("Weighted average smoothing for Gaussian noise reduction")
                st.image(numpy_to_streamlit(results['gaussian']), use_container_width=True)
            with col2:
                st.markdown("**Median Blur**")
                st.caption("Non-linear filter excellent for salt-and-pepper noise")
                st.image(numpy_to_streamlit(results['median']), use_container_width=True)
            
            # Row 4: Binarization
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Otsu's Thresholding**")
                st.caption("Automatic global threshold using bimodal histogram")
                st.image(numpy_to_streamlit(results['otsu']), use_container_width=True)
            with col2:
                st.markdown("**Adaptive Thresholding**")
                st.caption("Local thresholding for varying illumination")
                st.image(numpy_to_streamlit(results['adaptive']), use_container_width=True)
            
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
            
            with col3:
                _, buffer = cv2.imencode('.png', results['adaptive'])
                st.download_button(
                    label="⬇️ Download Adaptive Binary",
                    data=buffer.tobytes(),
                    file_name=f"page_{page_index + 1}_adaptive.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    main()

import streamlit as st
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Disable file watcher
os.environ["STREAMLIT_WATCHDOG_MONITOR_OWN_CHANGES"] = "false"

# Try importing OpenCV with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.error("OpenCV could not be imported. Some features may be limited.")

# Try importing PyTorch with fallback
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("PyTorch or YOLO could not be imported. Some features may be limited.")

# Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .stats-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stat-item {
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #757575;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .error-msg {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model(model_path):
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available. Cannot load model.")
        return None
        
    try:
        # Add safe globals for ultralytics
        import torch.serialization
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Load model with explicit task and device
        model = YOLO(model_path)
        
        # Move model to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Verify model loaded correctly
        if not hasattr(model, 'model'):
            raise AttributeError("Model loaded but missing required attributes")
            
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please ensure you have the correct model file and it's not corrupted.")
        return None

def get_detection_stats(results):
    """Extract statistics from detection results"""
    total_objects = 0
    helmet_count = 0
    no_helmet_count = 0
    bbox_data = []
    
    for r in results:
        total_objects += len(r.boxes)
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy())  # Class index
            conf = float(box.conf.cpu().numpy())  # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box
            
            if cls == 0:  # Helmet
                helmet_count += 1
            else:  # No Helmet
                no_helmet_count += 1
            
            bbox_data.append([cls, conf, x1, y1, x2, y2])
    
    return total_objects, helmet_count, no_helmet_count, bbox_data

def process_image(model, image, confidence_threshold, nms_threshold):
    if not CV2_AVAILABLE or not TORCH_AVAILABLE:
        st.error("Required dependencies are not available.")
        return None, 0, 0, 0, [], 0
        
    try:
        start_time = time.time()
        
        # Convert image to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure image is in correct format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
        # Run inference
        results = model(image, conf=confidence_threshold, iou=nms_threshold)
        infer_time = time.time() - start_time
        
        # Get detection stats
        total_objects, helmet_count, no_helmet_count, bbox_data = get_detection_stats(results)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        return annotated_img, total_objects, helmet_count, no_helmet_count, bbox_data, infer_time
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, 0, 0, 0, [], 0

def process_video(model, video_path, confidence_threshold, nms_threshold, progress_bar=None):
    if not CV2_AVAILABLE or not TORCH_AVAILABLE:
        st.error("Required dependencies are not available.")
        return None, 0, 0, 0, 0
        
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
        
        total_objects, helmet_count, no_helmet_count = 0, 0, 0
        start_time = time.time()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=confidence_threshold, iou=nms_threshold)
            t_obj, h_count, nh_count, _ = get_detection_stats(results)
            total_objects += t_obj
            helmet_count += h_count
            no_helmet_count += nh_count
            
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count += 1
            if progress_bar is not None:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        cap.release()
        out.release()
        infer_time = time.time() - start_time
        
        return temp_video.name, total_objects, helmet_count, no_helmet_count, infer_time
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, 0, 0, 0, 0

def main():
    # Sidebar settings
    st.sidebar.markdown('<div class="sub-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Model selection with better error handling
    available_models = []
    if os.path.exists("models"):
        available_models = [f for f in os.listdir("models") if f.endswith(('.pt', '.pth'))]
    
    if available_models:
        model_name = st.sidebar.selectbox("Select Model", available_models, index=0)
        model_path = os.path.join("models", model_name)
    else:
        st.sidebar.warning("No model files found in 'models' directory")
        model_path = st.sidebar.text_input("Model Path", "models/best.pt")
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file not found at: {model_path}")
            st.sidebar.info("Please ensure you have a valid YOLO model file (.pt or .pth) in the models directory")
            return

    # Add confidence and NMS threshold settings
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum probability to filter weak detections"
    )
    
    nms_threshold = st.sidebar.slider(
        "NMS Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4, 
        step=0.05,
        help="Non-maximum suppression threshold"
    )
    
    # Load the model with better error handling
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load model. Please check the model path and ensure you have a valid YOLO model file.")
        st.info("""
        Troubleshooting steps:
        1. Make sure you have a valid YOLO model file (.pt or .pth)
        2. Check if the model file is in the correct location
        3. Verify you have the latest ultralytics package installed
        4. Try running: pip install -U ultralytics
        """)
        return
    
    # Main content
    st.markdown('<div class="main-header">🪖 Helmet Detection System</div>', unsafe_allow_html=True)
    st.markdown("Upload an image/video to detect helmets!")
    
    # Input selection tabs
    tab1, tab2 = st.tabs(["📸 Image", "🎬 Video"])
    
    # Image Tab
    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                img = np.array(img)
                
                with st.spinner("Processing image..."):
                    result_img, total_objects, helmet_count, no_helmet_count, bbox_data, infer_time = process_image(
                        model, img, confidence_threshold, nms_threshold
                    )
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                
                with col2:
                    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">📊 Total Objects: {total_objects}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">🪖 Helmets: {helmet_count}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">⚠️ No Helmets: {no_helmet_count}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">⏱️ Processing Time: {infer_time:.2f}s</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if bbox_data:
                    st.markdown('<div class="sub-header">📋 Detection Details</div>', unsafe_allow_html=True)
                    df = pd.DataFrame(bbox_data, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
                    df["Class"] = df["Class"].map({0: "Helmet", 1: "No Helmet"})
                    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Detection Results",
                        csv,
                        f"helmet_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key='download-csv'
                    )
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Video Tab
    with tab2:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            try:
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video.write(uploaded_video.read())
                
                st.video(temp_video.name)
                
                if st.button("Process Video"):
                    progress_bar = st.progress(0)
                    with st.spinner("Processing video..."):
                        processed_video, total_objects, helmet_count, no_helmet_count, infer_time = process_video(
                            model, temp_video.name, confidence_threshold, nms_threshold, progress_bar
                        )
                    
                    st.success("Video processing complete!")
                    st.video(processed_video)
                    
                    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">📊 Total Objects: {total_objects}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">🪖 Helmets: {helmet_count}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">⚠️ No Helmets: {no_helmet_count}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-item">⏱️ Processing Time: {infer_time:.2f}s</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Provide download link for the processed video
                    with open(processed_video, "rb") as file:
                        st.download_button(
                            label="Download Processed Video",
                            data=file,
                            file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"Error processing video: {e}")
    
    # Footer
    st.markdown('<div class="footer">Helmet Detection System © 2025 | Made with ❤️ using YOLOv8 and Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
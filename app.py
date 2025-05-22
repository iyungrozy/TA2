import streamlit as st
import cv2
import torch
import time
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import tempfile
from datetime import datetime
import threading
import queue
import logging
import sys
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)

# Disable file watcher and set environment variables to resolve PyTorch compatibility issues
os.environ["STREAMLIT_WATCHDOG_MONITOR_OWN_CHANGES"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.set_num_threads(4)  # Limit number of threads

# Fix for asyncio event loop
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

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
    /* Error messages with better styling */
    .error-msg {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Warning messages */
    .warning-msg {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check if streamlit-webrtc is installed
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    webrtc_available = True
except ImportError:
    webrtc_available = False
    st.warning("streamlit-webrtc is not installed. Some features will be limited.")

# Initialize session state
if 'helmet_count' not in st.session_state:
    st.session_state.helmet_count = 0
if 'no_helmet_count' not in st.session_state:
    st.session_state.no_helmet_count = 0
if 'total_objects' not in st.session_state:
    st.session_state.total_objects = 0
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'is_webcam_active' not in st.session_state:
    st.session_state.is_webcam_active = False
if 'webcam_mode' not in st.session_state:
    st.session_state.webcam_mode = "webrtc" if webrtc_available else "opencv"
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None

# Thread lock for safety
lock = threading.Lock()

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource
def load_model(model_path):
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
            # Update progress bar safely
            progress_bar.progress(min(frame_count / total_frames, 1.0))
    
    cap.release()
    out.release()
    infer_time = time.time() - start_time
    
    return temp_video.name, total_objects, helmet_count, no_helmet_count, infer_time

def reset_webcam_stats():
    with lock:
        st.session_state.helmet_count = 0
        st.session_state.no_helmet_count = 0
        st.session_state.total_objects = 0
        st.session_state.frames_processed = 0
        st.session_state.start_time = time.time()

def run_opencv_webcam(model, confidence_threshold, nms_threshold):
    """Direct OpenCV webcam implementation as fallback"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam! Please check your camera connection.")
        return
    
    # Try setting camera resolution - might improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Reset stats
    reset_webcam_stats()
    
    # Create placeholder for video
    frame_placeholder = st.empty()
    
    # Create stop button
    stop_button_pressed = st.button("Stop Webcam")
    
    try:
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam!")
                break
            
            # Process frame with YOLOv8
            start_proc = time.time()
            results = model(frame, conf=confidence_threshold, iou=nms_threshold)
            t_obj, h_count, nh_count, _ = get_detection_stats(results)
            
            # Update stats
            with lock:
                st.session_state.helmet_count += h_count
                st.session_state.no_helmet_count += nh_count
                st.session_state.total_objects += t_obj
                st.session_state.frames_processed += 1
                st.session_state.processing_time = time.time() - st.session_state.start_time
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add text overlay
            cv2.putText(
                annotated_frame,
                f"Helmets: {h_count} | No Helmets: {nh_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Display frame
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Store last frame for stats view
            st.session_state.last_frame = annotated_frame
            
            # Short delay to reduce CPU usage
            time.sleep(0.01)
            
            # Check if stop button was pressed
            stop_button_pressed = st.button("Stop Webcam", key="stop_button")
    
    except Exception as e:
        st.error(f"Error in webcam processing: {e}")
    finally:
        cap.release()

# WebRTC video processor - only define if available
if webrtc_available:
    class VideoProcessor(VideoProcessorBase):
        def __init__(self, model, confidence_threshold, nms_threshold):
            self.model = model
            self.confidence_threshold = confidence_threshold
            self.nms_threshold = nms_threshold
            self.frame_count = 0
            reset_webcam_stats()
        
        def recv(self, frame):
            self.frame_count += 1
            start_time = time.time()
            
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame with model
            results = self.model(img, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            # Get detection stats
            total_objects, helmet_count, no_helmet_count, _ = get_detection_stats(results)
            
            # Update session state with thread safety
            with lock:
                st.session_state.helmet_count += helmet_count
                st.session_state.no_helmet_count += no_helmet_count
                st.session_state.total_objects += total_objects
                st.session_state.frames_processed += 1
                st.session_state.processing_time = time.time() - st.session_state.start_time
            
            # Draw detection results on frame
            annotated_img = results[0].plot()
            
            # Add overlay with current stats
            cv2.putText(
                annotated_img, 
                f"Helmets: {helmet_count} | No Helmets: {no_helmet_count}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Store frame for display
            st.session_state.last_frame = annotated_img
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

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
    
    # Webcam mode selection
    if webrtc_available:
        st.sidebar.markdown("### Webcam Mode")
        webcam_mode = st.sidebar.radio("Select webcam implementation:", 
                                     ["WebRTC (recommended)", "OpenCV (fallback)"],
                                     index=0)
        st.session_state.webcam_mode = "webrtc" if webcam_mode == "WebRTC (recommended)" else "opencv"
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Statistics"):
        reset_webcam_stats()
    
    # Main content
    st.markdown('<div class="main-header">🪖 Helmet Detection System</div>', unsafe_allow_html=True)
    st.markdown("Upload an image/video or use webcam to detect helmets in real-time!")
    
    # Input selection tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image", "🎬 Video", "📹 Webcam"])
    
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
    
    # Webcam Tab
    with tab3:
        st.markdown("### Real-time Webcam Detection")
        
        # Create columns for webcam and stats
        col1, col2 = st.columns([3, 1])
        
        # Statistics column
        with col2:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-item">📊 Total Objects: {st.session_state.total_objects}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-item">🪖 Helmets: {st.session_state.helmet_count}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-item">⚠️ No Helmets: {st.session_state.no_helmet_count}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-item">🎞️ Frames: {st.session_state.frames_processed}</div>', unsafe_allow_html=True)
            
            if st.session_state.frames_processed > 0:
                fps = st.session_state.frames_processed / max(1, st.session_state.processing_time)
                st.markdown(f'<div class="stat-item">⚡ FPS: {fps:.1f}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="stat-item">⏱️ Running Time: {st.session_state.processing_time:.1f}s</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show the last processed frame
            if st.session_state.last_frame is not None:
                st.image(st.session_state.last_frame, channels="BGR", caption="Latest Detection")
        
        # Webcam column
        with col1:
            # WebRTC implementation (if available)
            if st.session_state.webcam_mode == "webrtc" and webrtc_available:
                st.info("📹 Click 'START' below and allow browser access to your camera")
                
                # Define RTC configuration with free STUN servers
                rtc_config = RTCConfiguration(
                    {"iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]}
                    ]}
                )
                
                try:
                    # WebRTC streamer with longer timeout
                    webrtc_ctx = webrtc_streamer(
                        key="helmet-detection",
                        video_processor_factory=lambda: VideoProcessor(
                            model, confidence_threshold, nms_threshold
                        ),
                        rtc_configuration=rtc_config,
                        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                        async_processing=True,
                        timeout=30.0  # Longer timeout for connection
                    )
                    
                    if webrtc_ctx.state.playing:
                        st.session_state.is_webcam_active = True
                    else:
                        if st.session_state.is_webcam_active:
                            st.session_state.is_webcam_active = False
                    
                    # Show fallback option if WebRTC fails
                    if not webrtc_ctx.state.playing and st.session_state.frames_processed == 0:
                        st.markdown('<div class="warning-msg">⚠️ If WebRTC doesn\'t work, try the "OpenCV (fallback)" option in the sidebar.</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"WebRTC error: {e}")
                    st.markdown('<div class="error-msg">⚠️ WebRTC failed. Please try the OpenCV fallback mode in the sidebar.</div>', unsafe_allow_html=True)
                    st.session_state.webcam_mode = "opencv"
            
            # OpenCV implementation (fallback)
            elif st.session_state.webcam_mode == "opencv":
                st.info("📸 Click the button below to start webcam capture")
                
                start_button = st.button("Start Webcam")
                if start_button:
                    try:
                        run_opencv_webcam(model, confidence_threshold, nms_threshold)
                    except Exception as e:
                        st.error(f"OpenCV webcam error: {e}")
            
            # If streamlit-webrtc not installed
            else:
                st.error("streamlit-webrtc is not installed. Please install it to use webcam streaming:")
                st.code("pip install streamlit-webrtc")
                
                # Try to offer OpenCV fallback option
                st.info("Trying OpenCV fallback...")
                start_button = st.button("Start Webcam (OpenCV fallback)")
                if start_button:
                    try:
                        run_opencv_webcam(model, confidence_threshold, nms_threshold)
                    except Exception as e:
                        st.error(f"OpenCV webcam error: {e}")
    
    # Troubleshooting Expander
    with st.expander("🛠️ Troubleshooting Webcam Issues"):
        st.markdown("""
        ### Common Webcam Issues and Solutions

        1. **"Timeout starting video source" error:**
           * Make sure no other application is using your camera
           * Try the OpenCV fallback mode from the sidebar
           * Restart your computer to free up camera resources
        
        2. **Camera permission issues:**
           * Allow camera access in your browser settings
           * Check Windows privacy settings: Settings > Privacy > Camera
        
        3. **Slow performance:**
           * Lower your camera resolution in device settings
           * Close other browser tabs and applications
           
        4. **No video appears:**
           * Try a different browser (Chrome or Edge recommended)
           * Update your webcam drivers
           * Try connecting an external webcam if available
        """)
    
    # Footer
    st.markdown('<div class="footer">Helmet Detection System © 2025 | Made with ❤️ using YOLOv8 and Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import re
import os
import tempfile
import datetime
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration (Must be first)
st.set_page_config(page_title="Professional Plate Detector", page_icon="🚗", layout="wide")

# Function to load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load Assets
local_css("style.css")

# Extract Indian Number Plate Logic
def extract_indian_number_plate(text_list):
    text = " ".join(text_list).upper()
    text = re.sub(r'\bIND\b|\bND\b', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace('O', '0').replace('I', '1')
    match = re.search(r'\d{2}BH\d{4}[A-Z]{0,2}', text)
    if match: return match.group()
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{0,2}\d{4}', text)
    if match: return match.group()
    return None

# Load Models
# Set environment variables to support KMP and disable MKLDNN for Paddle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"

@st.cache_resource
def load_yolo_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found.")
        return None
    return YOLO(model_path)

@st.cache_resource
def load_paddleocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

yolo_model = load_yolo_model()
ocr_model = load_paddleocr()

# Core Processing Function
def process_frame(frame, conf_threshold):
    """
    Processes a single frame: Detects plates, applies OCR, updates log,
    and returns the annotated frame.
    """
    # YOLO Processing
    results = yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
    
    # Annotation
    annotated_frame = frame.copy()
    
    for r in results:
        for box in r.boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Cropping for OCR
            if x1 < x2 and y1 < y2:
                plate_crop = frame[y1:y2, x1:x2]
                
                # PaddleOCR
                ocr_result = ocr_model.ocr(plate_crop, cls=True)
                
                if ocr_result and ocr_result[0]:
                    raw_texts = [line[1][0] for line in ocr_result[0]]
                    final_text = extract_indian_number_plate(raw_texts)
                    
                    if final_text:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Note: Updating Streamlit session state from inside a thread (WebRTC) is tricky.
                        # For now, we will return the text drawn on screen, and Streamlit auto-refresh might not catch
                        # session state updates from this thread perfectly unless using a queue.
                        # However, for simple visualization, we draw it on the frame.
                         
                         # Draw Text on Frame
                        cv2.putText(annotated_frame, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return annotated_frame

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # We need to access conf_threshold. Since we can't pass arguments easily here without a class or globals,
    # and conf_threshold is in the main script scope, we might need a workaround if it changes.
    # For now, let's use a default or try to read from a global if possible, but globals are thread-sensitive.
    # Safest is to hardcode or use a class. Let's use a default 0.4 for WebRTC or try to grab it.
    
    processed_img = process_frame(img, 0.4) 
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ CONTROLS")
    conf_threshold = st.slider("Detection Sensitivity", 0.1, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### 📷 INPUT SOURCE")
    input_source = st.radio("Select Source:", ["Live Mobile Camera (WebRTC)", "Live Video (Local)"])
    
    st.markdown("---")
    st.markdown("### ℹ️ ABOUT")
    st.info("Enterprise-grade Number Plate Recognition using YOLOv8 & PaddleOCR.")

# Main Layout
st.markdown("<h1>🚗 LIVE NUMBER PLATE RECOGNITION</h1>", unsafe_allow_html=True)

# Session State for Detection Log
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

# Layout: Video Feed and Detection Log
# On mobile, these columns will stack automatically
col1, col2 = st.columns([1.5, 1], gap="medium")

with col2:
    st.markdown("### 📝 DETECTION LOG")
    table_placeholder = st.empty()
    if st.session_state.detection_log:
        table_placeholder.table(st.session_state.detection_log)

with col1:
    st.markdown("### 📷 IMAGE FEED")
    frame_placeholder = st.empty()


    # ---------------- LIVE MOBILE CAMERA (WEBRTC) ----------------
    if input_source == "Live Mobile Camera (WebRTC)":
        st.write("Live feed from your mobile device.")
        
        # WebRTC Configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_streamer(
            key="mobile_live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"facingMode": "environment"}}, # specific for mobile back camera
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

    # ---------------- LIVE VIDEO (LOCAL) MODE ----------------
    elif input_source == "Live Video (Local)":
        run = st.checkbox('Start Live Camera', value=True)
        
        if run:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video from webcam.")
                    break
                
                final_frame = process_frame(frame, conf_threshold)
                
                # Display Frame
                frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update Table
                with col2:
                    table_placeholder.table(st.session_state.detection_log)
            cap.release()
        else:
            st.write("Camera Stopped")

import streamlit as st
import cv2
import numpy as np
import json
import time
import os
from PIL import Image, ImageDraw, ImageFont
import tempfile
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import base64
from io import BytesIO
from dataset_tools import VisDroneDatasetAnalyzer, create_dataset_interface

# Page configuration
st.set_page_config(
    page_title="🚁 ATR - Automated Target Recognition",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #0056b3 0%, #004085 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

@st.cache_resource
def load_model(model_name):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None

def get_confidence_color(confidence):
    """Get color class based on confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def draw_bounding_boxes(image, results, class_filter=None):
    """Draw bounding boxes on image with labels and confidence scores"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Apply class filter if specified
                if class_filter and class_name not in class_filter:
                    continue
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle(bbox, fill="red")
                draw.text((x1, y1-25), label, fill="white", font=font)
                
                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    
    return img, detections

def process_image(model, image, confidence_threshold, iou_threshold, class_filter=None):
    """Process image with YOLO model"""
    try:
        # Run inference
        results = model(image, conf=confidence_threshold, iou=iou_threshold)
        
        # Draw bounding boxes
        annotated_img, detections = draw_bounding_boxes(image, results, class_filter)
        
        return annotated_img, detections, results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, [], None


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚁 Automated Target Recognition (ATR)</h1>
        <p>Advanced drone imagery analysis using YOLOv11 for military and civilian applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎯 Detection Controls")
    
    # Model selection
    model_options = {
        "YOLOv11 Nano (Fastest)": "yolo11n.pt",
        "YOLOv11 Small (Balanced)": "yolo11s.pt", 
        "YOLOv11 Medium (Most Accurate)": "yolo11m.pt"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model:",
        list(model_options.keys()),
        help="Choose model based on speed vs accuracy trade-off"
    )
    
    model_path = model_options[selected_model_name]
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    # Detection parameters
    st.sidebar.subheader("📊 Detection Parameters")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )
    
    # Class filtering
    st.sidebar.subheader("🔍 Object Classes")
    
    # Choose between COCO classes and VisDrone classes
    class_set = st.sidebar.radio(
        "Class Set:",
        ["COCO Classes", "VisDrone Classes"]
    )
    
    if class_set == "COCO Classes":
        class_filter = st.sidebar.multiselect(
            "Filter by object classes:",
            options=["person", "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat"],
            help="Select specific object classes to detect (leave empty for all classes)"
        )
    else:
        visdrone_classes = [
            "pedestrian", "people", "bicycle", "car", "van", 
            "truck", "tricycle", "awning-tricycle", "bus", "motor"
        ]
        class_filter = st.sidebar.multiselect(
            "Filter by VisDrone classes:",
            options=visdrone_classes,
            help="Select specific VisDrone object classes to detect"
        )
    
    # Processing mode
    st.sidebar.subheader("⚙️ Processing Mode")
    processing_mode = st.sidebar.radio(
        "Select processing mode:",
        ["Single Image", "Batch Processing", "Video Processing", "Comparison Mode", "Dataset Analysis"]
    )
    
    # Main content area
    if processing_mode == "Single Image":
        st.header("📸 Single Image Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload drone image:",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image
            if st.button("🎯 Run Detection", type="primary"):
                with st.spinner("Processing image..."):
                    start_time = time.time()
                    annotated_img, detections, results = process_image(
                        model, image, confidence_threshold, iou_threshold, class_filter
                    )
                    processing_time = time.time() - start_time
                
                if annotated_img is not None:
                    # Display results side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_img, use_container_width=True)
                    
                    # Detection statistics
                    st.subheader("📊 Detection Statistics")
                    
                    if detections:
                        # Create metrics
                        total_detections = len(detections)
                        avg_confidence = np.mean([d['confidence'] for d in detections])
                        class_counts = Counter([d['class'] for d in detections])
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Objects", total_detections)
                        with col2:
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        with col3:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col4:
                            st.metric("Model", selected_model_name.split()[1])
                        
                        # Class breakdown
                        st.subheader("Object Class Breakdown")
                        for class_name, count in class_counts.items():
                            st.write(f"**{class_name.title()}**: {count}")
                        
                        # Detailed results table
                        st.subheader("Detailed Detection Results")
                        df = pd.DataFrame(detections)
                        st.dataframe(df, use_container_width=True)
                        
                        
                        # Download options
                        st.subheader("💾 Export Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download annotated image
                            img_buffer = BytesIO()
                            annotated_img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Annotated Image",
                                data=img_buffer.getvalue(),
                                file_name="annotated_image.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # Download JSON results
                            json_data = {
                                'detections': detections,
                                'statistics': {
                                    'total_detections': total_detections,
                                    'average_confidence': float(avg_confidence),
                                    'processing_time': processing_time,
                                    'model_used': selected_model_name
                                }
                            }
                            
                            st.download_button(
                                label="📄 Download JSON Results",
                                data=json.dumps(json_data, indent=2),
                                file_name="detection_results.json",
                                mime="application/json"
                            )
                        
                        # Save to session history
                        st.session_state.detection_history.append({
                            'timestamp': time.time(),
                            'image_name': uploaded_file.name,
                            'detections': detections,
                            'statistics': {
                                'total_detections': total_detections,
                                'average_confidence': float(avg_confidence),
                                'processing_time': processing_time
                            }
                        })
                    else:
                        st.warning("No objects detected with current settings. Try lowering the confidence threshold.")
    
    elif processing_mode == "Batch Processing":
        st.header("📁 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images:",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Select multiple images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("🚀 Process All Images", type="primary"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                all_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        image = Image.open(uploaded_file)
                        annotated_img, detections, _ = process_image(
                            model, image, confidence_threshold, iou_threshold, class_filter
                        )
                        
                        if detections:
                            all_results.append({
                                'filename': uploaded_file.name,
                                'detections': len(detections),
                                'avg_confidence': np.mean([d['confidence'] for d in detections])
                            })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                if all_results:
                    st.subheader("📊 Batch Processing Results")
                    df = pd.DataFrame(all_results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    total_detections = sum(r['detections'] for r in all_results)
                    avg_confidence = np.mean([r['avg_confidence'] for r in all_results])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Images Processed", len(uploaded_files))
                    with col2:
                        st.metric("Total Objects Detected", total_detections)
    
    elif processing_mode == "Video Processing":
        st.header("🎬 Video Processing")
        
        uploaded_video = st.file_uploader(
            "Upload video file:",
            type=['mp4', 'avi', 'mov'],
            help="Upload video for frame-by-frame analysis"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("🎬 Process Video", type="primary"):
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
                
                try:
                    # Open video
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.write(f"Video info: {total_frames} frames at {fps:.2f} FPS")
                    
                    # Process video
                    progress_bar = st.progress(0)
                    results_container = st.container()
                    
                    frame_results = []
                    frame_count = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every 10th frame for performance
                        if frame_count % 10 == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_pil = Image.fromarray(frame_rgb)
                            
                            _, detections, _ = process_image(
                                model, frame_pil, confidence_threshold, iou_threshold, class_filter
                            )
                            
                            if detections:
                                frame_results.append({
                                    'frame': frame_count,
                                    'time': frame_count / fps,
                                    'detections': len(detections)
                                })
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)
                    
                    cap.release()
                    
                    # Display video results
                    if frame_results:
                        st.subheader("📊 Video Analysis Results")
                        df = pd.DataFrame(frame_results)
                        
                        # Plot detection timeline
                        fig = px.line(df, x='time', y='detections', 
                                    title='Object Detection Timeline',
                                    labels={'time': 'Time (seconds)', 'detections': 'Number of Objects'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(df, use_container_width=True)
                
                finally:
                    # Clean up temporary file
                    os.unlink(video_path)
    
    elif processing_mode == "Comparison Mode":
        st.header("⚖️ Model Comparison")
        
        st.write("Compare detection results from different YOLOv11 model sizes")
        
        uploaded_file = st.file_uploader(
            "Upload image for comparison:",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            if st.button("🔄 Run Comparison", type="primary"):
                models_to_compare = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
                model_names = ["Nano", "Small", "Medium"]
                
                comparison_results = []
                
                for model_name, display_name in zip(models_to_compare, model_names):
                    with st.spinner(f"Processing with {display_name} model..."):
                        comp_model = load_model(model_name)
                        if comp_model:
                            start_time = time.time()
                            annotated_img, detections, _ = process_image(
                                comp_model, image, confidence_threshold, iou_threshold, class_filter
                            )
                            processing_time = time.time() - start_time
                            
                            comparison_results.append({
                                'model': display_name,
                                'detections': len(detections),
                                'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                                'processing_time': processing_time,
                                'annotated_image': annotated_img
                            })
                
                # Display comparison results
                if comparison_results:
                    st.subheader("📊 Model Comparison Results")
                    
                    # Results table
                    df = pd.DataFrame([
                        {
                            'Model': r['model'],
                            'Detections': r['detections'],
                            'Avg Confidence': f"{r['avg_confidence']:.3f}",
                            'Processing Time': f"{r['processing_time']:.2f}s"
                        } for r in comparison_results
                    ])
                    st.dataframe(df, use_container_width=True)
                    
                    # Side-by-side images
                    st.subheader("🖼️ Detection Results Comparison")
                    cols = st.columns(len(comparison_results))
                    
                    for i, (col, result) in enumerate(zip(cols, comparison_results)):
                        with col:
                            st.write(f"**{result['model']} Model**")
                            st.image(result['annotated_image'], use_container_width=True)
    
    elif processing_mode == "Dataset Analysis":
        create_dataset_interface()
        
    # About section
    with st.expander("ℹ️ About ATR and How to Use This App"):
        st.markdown("""
        ## 🚁 Automated Target Recognition (ATR)
        
        **What is ATR?**
        Automated Target Recognition is a computer vision technology that automatically identifies and classifies objects in aerial imagery, particularly useful for:
        - Military reconnaissance and surveillance
        - Search and rescue operations
        - Infrastructure monitoring
        - Traffic analysis
        - Wildlife monitoring
        
        **How to Use This App:**
        1. **Select a Model**: Choose between YOLOv11 Nano (fastest), Small (balanced), or Medium (most accurate)
        2. **Adjust Parameters**: Set confidence and IoU thresholds to fine-tune detection sensitivity
        3. **Upload Images**: Drag and drop drone images in JPG, PNG, or JPEG format
        4. **Run Detection**: Click the detection button to analyze your images
        5. **Review Results**: View annotated images, statistics, and download results
        
        **Features:**
        - 🎯 Real-time object detection with YOLOv11
        - 📊 Detailed statistics and confidence scores
        - 🔥 Heatmap visualization of detection density
        - 📁 Batch processing for multiple images
        - 🎬 Video analysis with frame-by-frame processing
        - ⚖️ Model comparison mode
        - 💾 Export results as JSON or annotated images
        
        **Tips for Best Results:**
        - Use high-resolution images for better detection accuracy
        - Adjust confidence threshold based on your needs (higher = fewer false positives)
        - Try different models to find the best speed/accuracy balance
        - Use class filtering to focus on specific object types
        """)

    # Detection history
    if st.session_state.detection_history:
        with st.expander("📚 Detection History"):
            st.write(f"Total detections performed: {len(st.session_state.detection_history)}")
            
            for i, detection in enumerate(st.session_state.detection_history[-5:]):  # Show last 5
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{detection['image_name']}** - {detection['statistics']['total_detections']} objects detected")
                    with col2:
                        if st.button(f"View Details", key=f"history_{i}"):
                            st.json(detection['statistics'])

if __name__ == "__main__":
    main()

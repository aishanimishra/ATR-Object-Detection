# 🚁 ATR - Automated Target Recognition

A comprehensive Streamlit web application for Automated Target Recognition (ATR) using YOLOv11 for drone imagery analysis.

## Features

### Core Functionality
- **Image Upload**: Support for JPG, PNG, JPEG formats
- **YOLOv11 Detection**: Real-time object detection with multiple model sizes
- **Side-by-side Display**: Original and annotated images
- **Bounding Boxes**: Labels and confidence scores on detected objects

### User Controls
- **Model Selection**: Choose between YOLOv11 Nano, Small, and Medium models
- **Confidence Threshold**: Adjustable slider (0.0 to 1.0, default 0.25)
- **IoU Threshold**: Detection threshold slider (0.0 to 1.0, default 0.45)
- **Class Filtering**: Filter by specific object classes

### Results Display
- **Two-column Layout**: Original image on left, annotated results on right
- **Detection Statistics**: Total objects, class breakdown, average confidence
- **Download Options**: Annotated images and JSON results
- **Color-coded Confidence**: Green (high), Yellow (medium), Red (low)

### Advanced Features
- **Batch Processing**: Multiple image uploads with progress tracking
- **Video Processing**: Frame-by-frame analysis with timeline visualization
- **Comparison Mode**: Side-by-side model comparison
- **Detection History**: Session-based detection tracking

### UI/UX Design
- **Modern Interface**: Clean, professional design with drone theme
- **Responsive Layout**: Works on different screen sizes
- **Loading Indicators**: Spinners and progress bars
- **Error Handling**: Graceful error management
- **Tooltips**: Helpful explanations for technical terms

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd atr-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv11 models** (optional - will download automatically on first use):
   ```bash
   # Models will be downloaded automatically when first used
   # yolo11n.pt, yolo11s.pt, yolo11m.pt
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Select processing mode**:
   - **Single Image**: Upload and analyze individual images
   - **Batch Processing**: Process multiple images at once
   - **Video Processing**: Analyze video files frame-by-frame
   - **Comparison Mode**: Compare different model sizes

4. **Configure detection parameters**:
   - Choose model size (Nano/Small/Medium)
   - Adjust confidence and IoU thresholds
   - Select object classes to detect

5. **Upload and analyze**:
   - Drag and drop images or videos
   - Click "Run Detection" to process
   - View results and download outputs

## Model Information

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv11 Nano | ~6MB | Fastest | Good | Real-time applications |
| YOLOv11 Small | ~22MB | Balanced | Better | General purpose |
| YOLOv11 Medium | ~50MB | Slower | Best | High accuracy needs |

## Supported Object Classes

The application can detect and classify:
- Person
- Vehicle (car, truck, bus, motorcycle, bicycle)
- Aircraft
- Boats
- And many more COCO dataset classes

## Technical Requirements

- Python 3.8+
- CUDA support (optional, for GPU acceleration)
- 4GB+ RAM recommended
- Web browser with JavaScript enabled

## Performance Tips

1. **For Real-time Processing**: Use YOLOv11 Nano model
2. **For High Accuracy**: Use YOLOv11 Medium model
3. **For Balanced Performance**: Use YOLOv11 Small model
4. **Batch Processing**: Process multiple images for efficiency
5. **Video Processing**: Consider processing every Nth frame for performance

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure internet connection for model downloads
   - Check available disk space
   - Verify PyTorch installation

2. **Memory Issues**:
   - Use smaller model (Nano instead of Medium)
   - Reduce image resolution
   - Process images individually instead of batch

3. **Performance Issues**:
   - Enable GPU acceleration if available
   - Use lower confidence thresholds
   - Process fewer frames in video mode

### Error Messages

- **"Failed to load model"**: Check model file exists and is valid
- **"No objects detected"**: Try lowering confidence threshold
- **"Memory error"**: Reduce batch size or use smaller model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11 implementation
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision operations
- [Plotly](https://plotly.com/) for interactive visualizations

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This application is designed for educational and research purposes. Ensure compliance with local regulations when using for surveillance or military applications.

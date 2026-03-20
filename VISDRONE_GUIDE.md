# 🚁 VisDrone Dataset Integration Guide

This guide will help you integrate your VisDrone dataset with the ATR (Automated Target Recognition) Streamlit application.

## 📁 Dataset Structure

Your VisDrone dataset should be organized as follows:

```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/    (6471 images)
│   └── labels/    (6471 .txt files)
├── VisDrone2019-DET-val/
│   ├── images/    (548 images)
│   └── labels/    (548 .txt files)
├── VisDrone2019-DET-test-dev/
│   ├── images/    (1610 images)
│   └── labels/    (1610 .txt files)
└── visdrone_config.yaml
```

## 🚀 Quick Setup

### 1. Run the Setup Script
```bash
python setup_visdrone.py
```

This will create the proper directory structure and configuration files.

### 2. Copy Your Dataset
Copy your VisDrone images and labels to the created directories:
- Place images in `VisDrone_Dataset/VisDrone2019-DET-train/images/`
- Place labels in `VisDrone_Dataset/VisDrone2019-DET-train/labels/`
- Repeat for validation and test sets

### 3. Start the Application
```bash
streamlit run app.py
```

## 🎯 New Features Added

### 1. Dataset Analysis Mode
- **Visualize Dataset Statistics**: View class distributions, image sizes, and object counts
- **Sample Image Display**: See annotated sample images from your dataset
- **Dataset Validation**: Check for missing labels or format issues
- **Interactive Charts**: Pie charts and histograms showing dataset composition


### 2. Enhanced Detection
- **VisDrone Class Support**: Detect VisDrone-specific classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Class Filtering**: Choose between COCO classes or VisDrone classes
- **Custom Model Loading**: Use your trained models for detection

## 📊 VisDrone Classes

The application now supports all 10 VisDrone classes:

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | pedestrian | Individual pedestrians |
| 1 | people | Groups of people |
| 2 | bicycle | Bicycles |
| 3 | car | Cars |
| 4 | van | Vans |
| 5 | truck | Trucks |
| 6 | tricycle | Tricycles |
| 7 | awning-tricycle | Tricycles with awnings |
| 8 | bus | Buses |
| 9 | motor | Motorcycles |

## 🔧 Configuration File

The `visdrone_config.yaml` file contains:

```yaml
# VisDrone Dataset Configuration
path: ./VisDrone_Dataset
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

# number of classes
nc: 10

# Class names
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

## 📈 Dataset Analysis Features

### Statistical Overview
- **Total Images**: Count of images in each split
- **Total Objects**: Count of all detected objects
- **Average Objects per Image**: Mean number of objects per image
- **Class Distribution**: Number of instances per class

### Visualizations
- **Class Distribution Bar Chart**: Shows frequency of each class
- **Objects per Image Histogram**: Distribution of object counts
- **Image Size Scatter Plot**: Distribution of image dimensions
- **Class Distribution Pie Chart**: Interactive pie chart of classes

### Sample Visualization
- **Annotated Samples**: View sample images with bounding boxes
- **Class Labels**: See how different classes are labeled
- **Quality Check**: Verify annotation quality

## 🔍 Dataset Validation

The application automatically validates your dataset for:
- **Missing Labels**: Images without corresponding label files
- **Invalid Format**: Malformed label files
- **Class ID Issues**: Invalid class IDs or out-of-range values
- **File Structure**: Proper directory organization

## 🚀 Usage Tips

### For Dataset Analysis:
1. Use the "Dataset Analysis" mode to understand your data
2. Check class distribution to identify imbalanced classes
3. Review sample images to verify annotation quality
4. Validate dataset before training


### For Detection:
1. Use trained models for better performance on VisDrone classes
2. Adjust confidence threshold based on your needs
3. Filter classes to focus on specific object types
4. Compare different model sizes for speed vs accuracy trade-offs

## 🐛 Troubleshooting

### Common Issues:

1. **"Dataset path not found"**
   - Ensure your dataset is in the correct directory structure
   - Check file paths in the configuration

2. **"Config file not found"**
   - Run `python setup_visdrone.py` to create the config file
   - Verify the config file is in the project root

3. **"Missing labels"**
   - Check that all images have corresponding .txt label files
   - Ensure label files are in YOLO format

4. **Training errors**
   - Verify dataset structure and file formats
   - Check available GPU memory for batch size
   - Ensure all dependencies are installed

### Performance Tips:

1. **For faster training**: Use smaller batch sizes and image sizes
2. **For better accuracy**: Use larger models and more epochs
3. **For memory issues**: Reduce batch size or image resolution
4. **For GPU acceleration**: Ensure CUDA is properly installed

## 📚 Additional Resources

- [VisDrone Dataset Paper](https://arxiv.org/abs/1804.07383)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Format Specification](https://roboflow.com/formats/yolo-darknet-txt)



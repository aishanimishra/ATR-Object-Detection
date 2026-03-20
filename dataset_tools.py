import os
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import glob

class VisDroneDatasetAnalyzer:
    def __init__(self, dataset_path, config_path):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.config = self.load_config()
        self.class_names = self.config['names']
        self.num_classes = self.config['nc']
    
    def load_config(self):
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics"""
        stats = {
            'train': self._get_split_stats('VisDrone2019-DET-train'),
            'val': self._get_split_stats('VisDrone2019-DET-val'),
            'test-dev': self._get_split_stats('VisDrone2019-DET-test-dev')
        }
        return stats
    
    def _get_split_stats(self, split_name):
        """Get statistics for a specific dataset split"""
        images_path = os.path.join(self.dataset_path, split_name, 'images')
        labels_path = os.path.join(self.dataset_path, split_name, 'labels')
        
        if not os.path.exists(images_path):
            return None
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_path, ext)))
        
        # Get label files
        label_files = glob.glob(os.path.join(labels_path, '*.txt'))
        
        # Analyze labels
        class_counts = Counter()
        total_objects = 0
        image_sizes = []
        objects_per_image = []
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                image_objects = 0
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_objects += 1
                            image_objects += 1
                
                objects_per_image.append(image_objects)
            except (PermissionError, IOError) as e:
                # Skip files that can't be read due to permissions or other issues
                print(f"Warning: Could not read {label_file}: {e}")
                objects_per_image.append(0)
                continue
            
            # Get corresponding image size
            image_name = os.path.splitext(os.path.basename(label_file))[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = os.path.join(images_path, image_name + ext)
                if os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            image_sizes.append(img.size)
                    except:
                        pass
                    break
        
        return {
            'num_images': len(image_files),
            'num_labels': len(label_files),
            'total_objects': total_objects,
            'class_counts': dict(class_counts),
            'objects_per_image': objects_per_image,
            'image_sizes': image_sizes,
            'avg_objects_per_image': np.mean(objects_per_image) if objects_per_image else 0
        }
    
    def visualize_dataset(self, split='train'):
        """Create visualizations for the dataset"""
        stats = self.get_dataset_stats()
        split_stats = stats.get(split)
        
        if not split_stats:
            st.error(f"Split '{split}' not found in dataset")
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'VisDrone Dataset Analysis - {split.upper()} Split', fontsize=16)
        
        # Class distribution
        class_counts = split_stats['class_counts']
        class_names = [self.class_names[i] for i in class_counts.keys()]
        counts = list(class_counts.values())
        
        axes[0, 0].bar(class_names, counts)
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Objects per image distribution
        objects_per_image = split_stats['objects_per_image']
        axes[0, 1].hist(objects_per_image, bins=30, alpha=0.7)
        axes[0, 1].set_title('Objects per Image Distribution')
        axes[0, 1].set_xlabel('Number of Objects')
        axes[0, 1].set_ylabel('Frequency')
        
        # Image size distribution
        image_sizes = split_stats['image_sizes']
        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            
            axes[1, 0].scatter(widths, heights, alpha=0.5)
            axes[1, 0].set_title('Image Size Distribution')
            axes[1, 0].set_xlabel('Width (pixels)')
            axes[1, 0].set_ylabel('Height (pixels)')
        
        # Summary statistics
        summary_text = f"""
        Total Images: {split_stats['num_images']}
        Total Objects: {split_stats['total_objects']}
        Avg Objects/Image: {split_stats['avg_objects_per_image']:.2f}
        Classes Present: {len(class_counts)}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Dataset Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_sample_visualization(self, split='train', num_samples=9):
        """Create sample image visualization with annotations"""
        images_path = os.path.join(self.dataset_path, split, 'images')
        labels_path = os.path.join(self.dataset_path, split, 'labels')
        
        if not os.path.exists(images_path):
            st.error(f"Split '{split}' not found in dataset")
            return
        
        # Get sample images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_path, ext)))
        
        sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Create grid of images with annotations
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Sample Images with Annotations - {split.upper()} Split', fontsize=16)
        
        for idx, image_path in enumerate(sample_images):
            if idx >= 9:
                break
            
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # Load image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Load corresponding labels
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_path, image_name + '.txt')
            
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Convert to pixel coordinates
                                img_width, img_height = image.size
                                x1 = int((x_center - width/2) * img_width)
                                y1 = int((y_center - height/2) * img_height)
                                x2 = int((x_center + width/2) * img_width)
                                y2 = int((y_center + height/2) * img_height)
                                
                                # Draw bounding box
                                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                                
                                # Draw class label
                                class_name = self.class_names.get(class_id, f'class_{class_id}')
                                draw.text((x1, y1-20), class_name, fill='red')
                except (PermissionError, IOError) as e:
                    # Skip files that can't be read
                    print(f"Warning: Could not read {label_path}: {e}")
            
            ax.imshow(image)
            ax.set_title(f'Image {idx+1}')
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(sample_images), 9):
            row, col = idx // 3, idx % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def validate_dataset(self):
        """Validate dataset integrity"""
        issues = []
        
        for split in ['train', 'val', 'test-dev']:
            split_path = f'VisDrone2019-DET-{split}'
            images_path = os.path.join(self.dataset_path, split_path, 'images')
            labels_path = os.path.join(self.dataset_path, split_path, 'labels')
            
            if not os.path.exists(images_path):
                issues.append(f"Images directory not found: {images_path}")
                continue
            
            if not os.path.exists(labels_path):
                issues.append(f"Labels directory not found: {labels_path}")
                continue
            
            # Check for missing labels
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(images_path, ext)))
            
            label_files = glob.glob(os.path.join(labels_path, '*.txt'))
            
            image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
            label_names = {os.path.splitext(os.path.basename(f))[0] for f in label_files}
            
            missing_labels = image_names - label_names
            if missing_labels:
                issues.append(f"Missing labels for {len(missing_labels)} images in {split}")
            
            # Check label format
            for label_file in label_files[:10]:  # Check first 10 files
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) < 5:
                                issues.append(f"Invalid label format in {label_file}, line {line_num+1}")
                                continue
                            
                            try:
                                class_id = int(parts[0])
                                if class_id < 0 or class_id >= self.num_classes:
                                    issues.append(f"Invalid class ID {class_id} in {label_file}, line {line_num+1}")
                            except ValueError:
                                issues.append(f"Invalid class ID format in {label_file}, line {line_num+1}")
                except (PermissionError, IOError) as e:
                    issues.append(f"Could not read label file {label_file}: {e}")
        
        return issues

def create_dataset_interface():
    """Create Streamlit interface for dataset analysis"""
    st.header("📊 VisDrone Dataset Analysis")
    
    # Dataset path input
    dataset_path = st.text_input(
        "Dataset Path:",
        value="./VisDrone_Dataset",
        help="Path to the VisDrone dataset directory"
    )
    
    config_path = st.text_input(
        "Config File Path:",
        value="./visdrone_config.yaml",
        help="Path to the YAML configuration file"
    )
    
    if st.button("🔍 Analyze Dataset"):
        if not os.path.exists(dataset_path):
            st.error(f"Dataset path not found: {dataset_path}")
            return
        
        if not os.path.exists(config_path):
            st.error(f"Config file not found: {config_path}")
            return
        
        try:
            analyzer = VisDroneDatasetAnalyzer(dataset_path, config_path)
            
            # Display dataset statistics
            st.subheader("📈 Dataset Statistics")
            
            # Show warning about potential permission issues
            st.info("ℹ️ If you encounter permission errors, some files may be skipped during analysis. This is normal on Windows systems.")
            stats = analyzer.get_dataset_stats()
            
            # Create summary table
            summary_data = []
            for split, split_stats in stats.items():
                if split_stats:
                    summary_data.append({
                        'Split': split.upper(),
                        'Images': split_stats['num_images'],
                        'Objects': split_stats['total_objects'],
                        'Avg Objects/Image': f"{split_stats['avg_objects_per_image']:.2f}",
                        'Classes': len(split_stats['class_counts'])
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Display visualizations
                split_choice = st.selectbox("Select split for detailed analysis:", list(stats.keys()))
                
                if split_choice and stats[split_choice]:
                    # Dataset visualization
                    fig = analyzer.visualize_dataset(split_choice)
                    st.pyplot(fig)
                    
                    # Sample visualization
                    if st.button("Show Sample Images"):
                        sample_fig = analyzer.create_sample_visualization(split_choice)
                        st.pyplot(sample_fig)
                    
                    # Class distribution pie chart
                    st.subheader("🥧 Class Distribution")
                    class_counts = stats[split_choice]['class_counts']
                    if class_counts:
                        class_names = [analyzer.class_names[i] for i in class_counts.keys()]
                        counts = list(class_counts.values())
                        
                        fig_pie = px.pie(values=counts, names=class_names, 
                                       title=f"Class Distribution - {split_choice.upper()}")
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            # Dataset validation
            st.subheader("✅ Dataset Validation")
            issues = analyzer.validate_dataset()
            
            if issues:
                st.warning("Dataset validation issues found:")
                for issue in issues:
                    st.write(f"⚠️ {issue}")
            else:
                st.success("✅ Dataset validation passed! No issues found.")
        
        except Exception as e:
            st.error(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    create_dataset_interface()

"""
Dataset Conversion Utilities
Helper functions to convert various dataset formats to lmms-eval compatible JSON format
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class DatasetConverter:
    """Base class for converting datasets to lmms-eval JSON format"""
    
    @staticmethod
    def to_json(data: List[Dict[str, Any]], output_path: str, indent: int = 2) -> None:
        """Save dataset to JSON format"""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({"data": data}, f, indent=indent)
        print(f"Dataset saved to {output_path}")


class DirectoryToDataset(DatasetConverter):
    """Convert directory structure to dataset"""
    
    @staticmethod
    def from_directory(
        image_dir: str,
        output_path: str,
        annotation_dir: Optional[str] = None,
        file_extensions: tuple = ('.jpg', '.jpeg', '.png')
    ) -> None:
        """
        Convert directory of images to dataset JSON
        
        Args:
            image_dir: Directory containing images
            output_path: Output JSON path
            annotation_dir: Optional directory with annotation files (same name as images)
            file_extensions: Supported image extensions
        """
        dataset = []
        image_dir = Path(image_dir)
        
        for i, image_file in enumerate(sorted(image_dir.glob('*'))):
            if image_file.suffix.lower() not in file_extensions:
                continue
            
            sample = {
                "id": f"sample_{i:05d}",
                "image_path": str(image_file.absolute()),
                "question": "Describe any anomalies in this image",
                "answer": "Add ground truth here"
            }
            
            # Try to load annotation if provided
            if annotation_dir:
                anno_file = Path(annotation_dir) / f"{image_file.stem}.txt"
                if anno_file.exists():
                    sample["answer"] = anno_file.read_text().strip()
            
            dataset.append(sample)
        
        DirectoryToDataset.to_json(dataset, output_path)
        print(f"Converted {len(dataset)} images from {image_dir}")


class CSVToDataset(DatasetConverter):
    """Convert CSV to dataset"""
    
    @staticmethod
    def from_csv(
        csv_path: str,
        output_path: str,
        image_col: str = "image_path",
        question_col: str = "question",
        answer_col: str = "answer",
        id_col: Optional[str] = None
    ) -> None:
        """
        Convert CSV to dataset JSON
        
        Args:
            csv_path: Path to CSV file
            output_path: Output JSON path
            image_col: Column name for image paths
            question_col: Column name for questions
            answer_col: Column name for answers
            id_col: Column name for IDs (auto-generated if None)
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas required. Install with: pip install pandas")
            return
        
        df = pd.read_csv(csv_path)
        dataset = []
        
        for idx, row in df.iterrows():
            sample = {
                "id": row[id_col] if id_col and id_col in df.columns else f"sample_{idx:05d}",
                "image_path": row[image_col],
                "question": row[question_col],
                "answer": row[answer_col]
            }
            
            # Add any extra columns as categories
            for col in df.columns:
                if col not in [image_col, question_col, answer_col, id_col]:
                    sample[f"{col}"] = row[col]
            
            dataset.append(sample)
        
        CSVToDataset.to_json(dataset, output_path)
        print(f"Converted {len(dataset)} rows from {csv_path}")


class COCOToDataset(DatasetConverter):
    """Convert COCO format to dataset"""
    
    @staticmethod
    def from_coco(
        coco_json_path: str,
        output_path: str,
        image_dir: str,
        use_captions: bool = True
    ) -> None:
        """
        Convert COCO format dataset to lmms-eval JSON
        
        Args:
            coco_json_path: Path to COCO annotations JSON
            output_path: Output JSON path
            image_dir: Directory containing images
            use_captions: Use captions as ground truth
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image and annotation indices
        images_by_id = {img['id']: img for img in coco_data['images']}
        annotations_by_image_id = {}
        for anno in coco_data['annotations']:
            image_id = anno['image_id']
            if image_id not in annotations_by_image_id:
                annotations_by_image_id[image_id] = []
            annotations_by_image_id[image_id].append(anno)
        
        dataset = []
        for image_id, image_info in images_by_id.items():
            image_path = os.path.join(image_dir, image_info['file_name'])
            
            if image_id in annotations_by_image_id:
                annotations = annotations_by_image_id[image_id]
                answers = [a['caption'] for a in annotations] if use_captions else [str(a) for a in annotations]
                answer = " ".join(answers[:1]) if answers else ""
            else:
                answer = ""
            
            sample = {
                "id": f"coco_{image_id}",
                "image_path": image_path,
                "question": "Describe this image in detail",
                "answer": answer
            }
            dataset.append(sample)
        
        COCOToDataset.to_json(dataset, output_path)
        print(f"Converted {len(dataset)} COCO images")


class VideoDatasetConverter(DatasetConverter):
    """Convert video datasets to dataset"""
    
    @staticmethod
    def from_video_directory(
        video_dir: str,
        output_path: str,
        annotation_file: Optional[str] = None,
        video_extensions: tuple = ('.mp4', '.avi', '.mov')
    ) -> None:
        """
        Convert directory of videos to dataset JSON
        
        Args:
            video_dir: Directory containing videos
            output_path: Output JSON path
            annotation_file: JSON file with video annotations
            video_extensions: Supported video extensions
        """
        annotations = {}
        if annotation_file:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        
        dataset = []
        video_dir = Path(video_dir)
        
        for i, video_file in enumerate(sorted(video_dir.glob('*'))):
            if video_file.suffix.lower() not in video_extensions:
                continue
            
            video_id = video_file.stem
            sample = {
                "id": video_id,
                "video_path": str(video_file.absolute()),
                "question": "Describe what happens in this video",
                "answer": annotations.get(video_id, "Add ground truth here")
            }
            dataset.append(sample)
        
        VideoDatasetConverter.to_json(dataset, output_path)
        print(f"Converted {len(dataset)} videos from {video_dir}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Convert directory of images
    # DirectoryToDataset.from_directory(
    #     image_dir="/path/to/images",
    #     output_path="dataset.json",
    #     annotation_dir="/path/to/annotations"
    # )
    
    # Example 2: Convert CSV
    # CSVToDataset.from_csv(
    #     csv_path="data.csv",
    #     output_path="dataset.json",
    #     image_col="image_path",
    #     question_col="question",
    #     answer_col="answer"
    # )
    
    # Example 3: Convert COCO format
    # COCOToDataset.from_coco(
    #     coco_json_path="instances.json",
    #     output_path="dataset.json",
    #     image_dir="/path/to/images"
    # )
    
    # Example 4: Convert video directory
    # VideoDatasetConverter.from_video_directory(
    #     video_dir="/path/to/videos",
    #     output_path="video_dataset.json",
    #     annotation_file="annotations.json"
    # )
    
    print("Dataset conversion utilities loaded")
    print("See docstrings for usage examples")

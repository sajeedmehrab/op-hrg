import json
import numpy as np
from PIL import Image
from pathlib import Path
import os
import json
from collections import defaultdict

class PartImageNetDataset:
    def __init__(self, json_path, images_dir):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to test.json file
            images_dir: Directory containing the images
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.images_dir = Path(images_dir)

        # create a mapping between image file_name and id
        self.image_id_to_filename = {
            img['id']: img['file_name']
            for img in self.data['images']
        }

        # create a mapping between category id and category name
        self.category_id_to_name = {
            cat['id']: cat['name']
            for cat in self.data['categories']
        }

        # merge annotations with same image_id and category_id
        self.merged_annotations = self._merge_annotations(self.data)
        # number of annotations 

        self.num_annotations = len(self.merged_annotations)

    def get_annotation(self, idx):
        """
        Get annotation for a given index.
        An annotation of an index corresponds to a particular object part in an image.
        We can have multiple occurrences of the same object part in an image, hence multiple bboxes and masks for that annotation

        Args:
            idx: The nth annotation to retrieve
            
        Returns:
            dict containing:    
                - image_filename of the nth annotation
                - image_filepath of the nth annotation
                - class_name: str in format "{object}'s {part}"
                - bboxes: list of bboxes in xyxy format
                - mask: binary segmentation mask as numpy array
                - masks: list of binary segmentation masks as numpy arrays
                - image: PIL Image object
        """

        if idx >= self.num_annotations:
            raise ValueError(f"Invalid index {idx} for dataset with {self.num_annotations} annotations")

        annotation = self.merged_annotations[idx]
        image_id = annotation['image_id']

        # get image filename from image_id
        filename = self.image_id_to_filename.get(image_id)
        if filename is None:
            raise ValueError(f"Image with ID {image_id} not found")
        
        supercategory = filename.split('_')[0]
        image_dir = os.path.join(self.images_dir, supercategory)

        # Get image path
        image_path = os.path.join(image_dir, filename)

        # load image for height and width
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # get the category id (class label) from annotation
        category_id = int(annotation['category_id'])

        # get class name from category id
        try:
            class_name = self.category_id_to_name[category_id]
            class_name_split = class_name.split()
            assert len(class_name_split) > 0, "Class name should have object and part"
            class_name_formatted = f"{class_name_split[0]}'s"
            for part in class_name_split[1:]:
                class_name_formatted += f" {part}"
        except Exception as e:
            class_name_formatted = None

        # convert bbox from coco xywh to xyxy format
        bboxes = self._extract_bboxes(annotation)
        
        # get list of masks 
        masks = self._create_masks(annotation, img_height, img_width)
        
        return {
            'image_filename': filename,
            'image_filepath': image_path,
            'class_name': class_name_formatted,
            'bboxes': bboxes,
            'masks': masks,
            'mask': self._combine_masks(masks) if masks is not None else None,
            'image': image,
            'id': annotation['id']
        }
    
    def _merge_annotations(self, data):
        """
        Merge annotations with the same image_id and category_id. The original dataset has separate annotation IDs for each bbox/segmentation of the same object part in an image. 
        For example, if an image has 2 boxes for car's wheel, there will be 2 separate annotations with different IDs but same image_id and category_id.

        Args:
            data: original data dictionary from JSON, which has separate annotation IDs for each bbox/segmentation

        Returns:
            List of dictionaries, each representing a merged annotation.
        """
        
        # dictionary to group annotations
        # key: (image_id, category_id), value: list of annotations
        grouped = defaultdict(list)
        
        for ann in data['annotations']:
            key = (ann['image_id'], ann['category_id'])
            grouped[key].append(ann)

        # create merged annotations to merge bboxes and segmentations
        merged_annotations = []
        new_id = 0
        
        for (image_id, category_id), anns in grouped.items():
            # Collect all bboxes, segmentations, and old IDs
            bboxes = []
            segmentations = []
            old_ids = []
            
            for ann in anns:
                old_ids.append(ann['id'])
                if ann['bbox']: 
                    bboxes.append(ann['bbox'])
                if ann['segmentation']:  
                    segmentations.append(ann['segmentation'])
            
            # create merged annotation
            merged_ann = {
                'id': new_id,
                'image_id': image_id,
                'category_id': category_id,
                'bboxes': bboxes,
                'segmentations': segmentations,
                'old_annotation_ids': old_ids
            }
            
            merged_annotations.append(merged_ann)
            new_id += 1

        return merged_annotations
    
    def _extract_bboxes(self, annotation):
        """
        Extract and convert bounding boxes from COCO format (xywh) to xyxy.
        """
        # The annotation structure in your JSON shows 'area' field
        # You need to check if there's a 'bbox' field in the actual data
        bboxes = annotation.get('bboxes', [])
        converted_bboxes = []
        for bbox in bboxes:
            converted_bbox = self._convert_bbox_format(bbox)
            if converted_bbox is not None:
                converted_bboxes.append(converted_bbox)
            
        if len(converted_bboxes) > 0:
            return converted_bboxes
        else:
            return None

    def _convert_bbox_format(self, bbox):
        """
        Convert bounding box from COCO format (xywh) to (x1, y1, x2, y2).
        """
        try:
            x, y, w, h = bbox
        except Exception as e:
            print(f"Error unpacking bbox: {e}")
            return None
        
        converted_bbox = (x, y, x + w, y + h)
        return [round(coord + 0.5) for coord in converted_bbox]

    def _create_masks(self, annotation, height, width):
        """
        Create binary segmentation masks in a list for all segmentations in the annotation.
        """
        
        # If segmentation data exists in COCO format
        segmentations = annotation.get('segmentations', [])
        masks = []
        try:
            for segmentation in segmentations:
                # Polygon format
                from PIL import ImageDraw
                mask_img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask_img)

                for seg in segmentation:
                    # Convert flat list to list of tuples
                    polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, fill=1) 
            
                mask = np.array(mask_img)
                masks.append(mask)
        except Exception as e:
            print(f"Error creating masks: {e}")
            return None

        if len(masks) > 0:
            return masks
        else:
            return None

    def _combine_masks(self, masks):
        """
        combine multiple binary masks into a single mask.
        """
        if not masks:
            return None
        
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = combined_mask | mask
        
        return combined_mask
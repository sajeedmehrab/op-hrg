import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

PART_CLASSNAME_TO_READABLENAME = {
    # directions
    "lwing": "left wing",
    "rwing": "right wing",
    "lleg": "left leg",
    "rleg": "right leg",
    "lfoot": "left foot",
    "rfoot": "right foot",
    "leye": "left eye",
    "reye": "right eye",
    "lear": "left ear",
    "rear": "right ear",
    "llleg": "left lower leg",
    "luleg": "left upper leg",
    "rlleg": "right lower leg",
    "ruleg": "right upper leg",
    "llarm": "left lower arm",
    "luarm": "left upper arm",
    "rlarm": "right lower arm",
    "ruarm": "right upper arm",
    "lhand": "left hand",
    "rhand": "right hand",
    "lebrow": "left eyebrow",
    "rebrow": "right eyebrow",
    "lfleg": "left front leg",
    "rfleg": "right front leg",
    "lfpa": "left front paw",
    "rfpa": "right front paw",
    "lbleg": "left back leg",
    "rbleg": "right back leg",
    "lbpa": "left back paw",
    "rbpa": "right back paw",
    "lfuleg": "left front upper leg",
    "lflleg": "left front lower leg",
    "rfuleg": "right front upper leg",
    "rflleg": "right front lower leg",
    "lbuleg": "left back upper leg",
    "lblleg": "left back lower leg",
    "rbuleg": "right back upper leg",
    "rblleg": "right back lower leg",
    "lfho": "left front hoof",
    "rfho": "right front hoof",
    "lbho": "left back hoof",
    "rbho": "right back hoof",

    # parts
    "head": "head",
    "torso": "torso",
    "neck": "neck",
    "tail": "tail",
    "beak": "beak",
    "nose": "nose",
    "mouth": "mouth",
    "muzzle": "muzzle",
    "hair": "hair",
    "ear": "ear",
    "eye": "eye",
    "horn": "horn",
    "screen": "screen",
    "roofside": "roof side",
    "frontside": "front side",
    "backside": "back side",
    "leftside": "left side",
    "rightside": "right side",
    "cap": "cap",
    "body": "body",
    "pot": "pot",
    "plant": "plant",

    # objects & accessories
    "saddle": "saddle",
    "handlebar": "handle bar",
    "chainwheel": "chain wheel",
    "fwheel": "front wheel",
    "bwheel": "back wheel",
    "wheel": "wheel",
    "headlight": "headlight",
    "window": "window",
    "door": "door",
    "fliplate": "front license plate",
    "bliplate": "back license plate",
    "mirror": "mirror",
    "leftmirror": "left mirror",
    "rightmirror": "right mirror",
    "lhorn": "left horn",
    "rhorn": "right horn",

    # train
    "hfrontside": "head front side",
    "hleftside": "head left side",
    "hrightside": "head right side",
    "hbackside": "head back side",
    "hroofside": "head roof side",
    "coach": "coach",
    "cfrontside": "coach front side",
    "cleftside": "coach left side",
    "crightside": "coach right side",
    "cbackside": "coach back side",
    "croofside": "coach roof side",

    # aeroplane
    "engine": "engine",
    "stern": "stern"
}

def get_readable_part_name(part_classname):
    """
        Get the readable part name from the part class name.
        Gets rid of numbers and underscores used for numbered part instances. For example, "lwing_1" becomes just "left wing".
        If the class name is not in the dictionary, will lead to a KeyError.
    """
    base_partname = part_classname.split("_")[0]
    return PART_CLASSNAME_TO_READABLENAME[base_partname]

def load_annotation(annot_file_name, annotations_path):
    """Load the .mat annotation file."""
    mat_data = scipy.io.loadmat(os.path.join(annotations_path, annot_file_name))
    return mat_data['anno']

def get_pascalpart_masks(annot_file_name, annotations_path, images_path):
    """
    Return the segmentation masks for a single image from an annotation file.
    """
    # Load annotation
    anno = load_annotation(annot_file_name, annotations_path)
    
    # Extract objects
    objects = anno[0][0]['objects'][0]
    
    # Load image
    img_file = os.path.splitext(annot_file_name)[0] + ".jpg"
    img = Image.open(os.path.join(images_path, img_file))
    
    img_array = np.array(img)
    """
    {
        'class_name':
        {
            'object_maps': [list of maps]
            'parts': 
            {
                'part_name': [list of maps] # Once again, ideal expectation is one unique part per image, but we keep list of maps in case one part is present multiple times
            }
        }
    }
    """

    anno_dict = {}
    
    for obj_idx, obj in enumerate(objects): # You can have the same class multiple time in objects
        class_ind = obj['class_ind'][0][0]
        class_name = obj['class'][0]
        class_mask = obj['mask']

        # Extract parts
        try:
            parts = obj['parts'][0]
        except IndexError:
            continue

        if class_name not in anno_dict:
            anno_dict[class_name] = {'object_maps': []}
        anno_dict[class_name]['object_maps'].append(class_mask)
        
        # "parts" key could already be present if the same object appears multiple times
        if 'parts' not in anno_dict[class_name]:
            anno_dict[class_name]['parts'] = {}
        for i, part in enumerate(parts):
            part_name = part['part_name'][0]
            part_name = get_readable_part_name(part_name) # Get the readable part name
            part_mask_data = part['mask']
            if part_name not in anno_dict[class_name]['parts']:
                anno_dict[class_name]['parts'][part_name] = []
            anno_dict[class_name]['parts'][part_name].append(part_mask_data)

    return anno_dict

def get_pascalpart_classes(annot_file_name, annotations_path):
    """
        Return just the class names for a single image from an annotation .m file.
        We want to get the object name and all parts asscociated with it.
        Dictionary that has object name as key and a list of parts as value
    """
    # Load annotation
    anno = load_annotation(annot_file_name, annotations_path)

    classes_dict = {}
    
    # Extract objects
    objects = anno[0][0]['objects'][0]
    
    for obj_idx, obj in enumerate(objects): 
        class_name = str(obj['class'][0])
        # Extract parts
        try:
            parts = obj['parts'][0]
        except IndexError:
            # This object has no parts, and therefore is not of interest to us
            continue

        if class_name not in classes_dict:
            classes_dict[class_name] = set()
        
        for i, part in enumerate(parts):
            part_name = str(part['part_name'][0])
            part_name = get_readable_part_name(part_name) # Get the readable part name
            classes_dict[class_name].add(part_name)

    # Convert set to list
    for class_name, parts in classes_dict.items():
        classes_dict[class_name] = list(parts)     

    return classes_dict

def visualize_example(annot_file_name, annotations_path, images_path):
    """
        Visualize an example image and its segmentation masks.
        The different parts will be shown in different colors. 
    """
    # Load annotation
    anno = load_annotation(annot_file_name, annotations_path)
    
    # Extract objects
    objects = anno[0][0]['objects'][0]
    
    # Load image
    img_file = os.path.splitext(annot_file_name)[0] + ".jpg"
    img = Image.open(os.path.join(images_path, img_file))
    
    # Create masks
    img_array = np.array(img)
    cls_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    inst_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    part_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    for obj_idx, obj in enumerate(objects):
        class_ind = obj['class_ind'][0][0]
        print(f"Class Index: {class_ind}")
        class_name = obj['class'][0]
        silh = obj['mask']
        cls_mask[silh > 0] = class_ind
        inst_mask[silh > 0] = obj_idx + 1
        
        # Extract parts
        parts = obj['parts'][0]
        for i, part in enumerate(parts):
            part_name = part['part_name'][0]
            print(f"Class: {class_name}, Part: {part_name}")
            part_mask_data = part['mask']
            part_mask[part_mask_data > 0] = i+1  # Assign a value for visualization
    
    # Plot the image and masks
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[1].imshow(cls_mask, cmap='jet')
    axes[1].set_title("Class Mask")
    axes[2].imshow(inst_mask, cmap='jet')
    axes[2].set_title("Instance Mask")
    axes[3].imshow(part_mask, cmap='jet')
    axes[3].set_title("Part Mask")
    plt.show()

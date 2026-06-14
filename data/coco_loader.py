import json
from pathlib import Path
from typing import Any, Dict, Tuple
from collections import defaultdict

def load_coco(path_to_split: str) -> Dict[str, Any]:
    """
    Input:
    path_to_split - path to a dataset split (e.g. dataset.coco/train)

    Returns
    A dictionary containing dataset split metadata, classes, image information, and object information

    Assumption:
    - COCO json file name is "_annotations.coco.json"
    """
    # Check if path exists
    json_file_path = Path(path_to_split) / "_annotations.coco.json"

    if json_file_path.exists():
        with open(json_file_path) as f:
            return json.load(f)
    else:
        raise FileNotFoundError("'_annotations.coco.json' not found")

def index_coco(data: Dict[str, Any], path_to_split: str) -> Tuple[dict, dict, dict]:
    '''
    Input:
    data dictionary (COCO JSON annotations file)
    Path to split containing JSON file in use by data

    Return:
    tuple containing dicts:
    - Category ID to name
    - ID to img
    - annotations by img
    '''
    category_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    
    id_to_image = {img["id"]: img for img in data["images"]} 
    for img in id_to_image.values():
        img["abs_path"] = str(Path(path_to_split) / img["file_name"])

    # Create a dictionary with default value empty list
    annotation_by_img = defaultdict(list)

    for annotation in data["annotations"]:
        annotation_by_img[annotation["image_id"]].append(annotation)

    return (category_id_to_name, id_to_image, annotation_by_img)
    
def load_dataset_splits(path_to_dataset: str) -> dict:
    '''
    Returns
    indexed splits dictionary

    indexed_splits:
    - train : (category_id_to_name, id_to_image, annotation_by_img) (required)
    - valid : (category_id_to_name, id_to_image, annotation_by_img) OR None 
    - test : (category_id_to_name, id_to_image, annotation_by_img) OR None
    '''

    dataset = Path(path_to_dataset)

    indexed_splits = {}

    if dataset.exists():
        for folder in ["train", "valid", "test"]:
            path_to_split = dataset / folder

            try:
                coco_json = load_coco(path_to_split)
                indexed_data = index_coco(coco_json, path_to_split)
                indexed_splits[folder] = indexed_data
            except FileNotFoundError:  
                # train split is required, valid and train split is optional
                if folder == "train":
                    raise FileNotFoundError("Error no train split. Please add a folder called \"train\" in dataset")
                if folder == "valid":
                    print("No validation split found")
                    indexed_splits[folder] = None
                if folder == "test":
                    print("No test split found")
                    indexed_splits[folder] = None
    else:
        raise FileNotFoundError("dataset not found")

    return indexed_splits
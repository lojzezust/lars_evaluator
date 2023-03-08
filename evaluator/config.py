import os
from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = "semantic"

# Dataset configuration
_C.DATASET = CN()
_C.DATASET.IMAGE_SUBDIR  = "images"
_C.DATASET.SEMANTIC_MASK_SUBDIR  = "semantic_masks"
_C.DATASET.PANOPTIC_MASK_SUBDIR  = "panoptic_masks"
_C.DATASET.PANOPTIC_ANN_FILE  = "panoptic_annotations.json"
_C.DATASET.SUBSET_LIST = "image_list.txt"

# Semantic segmentation: class configuration
_C.SEGMENTATION = CN()
_C.SEGMENTATION.IDS = [0,1,2]
_C.SEGMENTATION.IGNORE_ID = 255
_C.SEGMENTATION.NAMES = ['obstacle', 'water', 'sky']
_C.SEGMENTATION.OBSTACLE_CLASS = 0
_C.SEGMENTATION.WATER_CLASS = 1
_C.SEGMENTATION.SKY_CLASS = 2
_C.SEGMENTATION.COLORS = [[247, 195,  37],  # Obstacles RGB color
                          [ 41, 167, 224],  # Water RGB color
                          [ 90,  75, 164]]  # Sky RGB color

# Panoptic segmentation: configuration
_C.PANOPTIC = CN()
_C.PANOPTIC.VOID_ID = 0 # Class ID for void predictions
_C.PANOPTIC.STATIC_OBSTACLE_CLASS = 1 # Class ID of static obstacles
_C.PANOPTIC.WATER_CLASS = 3 # Class ID of water
_C.PANOPTIC.SKY_CLASS = 5 # Class ID of sky
_C.PANOPTIC.DYN_OBST_IDS = [11,12,13,14,15,16,17,19] # IDs that count as dynamic obstacles
_C.PANOPTIC.IOU_THRESH = 0.5 # Normal IoU threshold

_C.PANOPTIC.SMALL_OBJECT_DIAG_THRESH = 10 # Threshold for objects to be treated as SMALL objects
_C.PANOPTIC.SMALL_OBJECT_IOU_THRESH = 0.5 # IoU threshold used on SMALL objects

# All Paths
_C.PATHS = CN()
_C.PATHS.RESULTS       = "./results"                # Path to where the results will be saved
_C.PATHS.DATASET_ROOT  = "/path/to/dataset"         # Path to where the dataset is stored
_C.PATHS.PREDICTIONS   = "/path/to/predictions/"    # Path to where the segmentation predictions are stored

# Evaluation configuration
_C.EVALUATION = CN()
_C.EVALUATION.MIN_COVERAGE = 0.7
_C.EVALUATION.WE_DILATION_SIZE = 11
_C.EVALUATION.FP_WATER_EROSION_SIZE = 21
_C.EVALUATION.SMALL_OBJECT_DILATION = 7

# Progress output configuration
_C.PROGRESS = CN()
_C.PROGRESS.METRICS = ['mIoU', 'WE_acc', 'F1']


def get_cfg(config_file=None):
    cfg = _C.clone()
    if config_file is not None:
       cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

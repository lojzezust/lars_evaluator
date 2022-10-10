import os
from yacs.config import CfgNode as CN

_C = CN()

# Dataset configuration
_C.DATASET = CN()
_C.DATASET.IMAGE_SUBDIR  = "images"
_C.DATASET.GT_MASK_SUBDIR  = "masks"
_C.DATASET.INST_MASK_SUBDIR  = "instances"
_C.DATASET.SUBSET_LIST = "list_test.txt"

# Class configuration
_C.CLASSES = CN()
_C.CLASSES.IDS = [0,1,2]
_C.CLASSES.IGNORE_ID = 4
_C.CLASSES.NAMES = ['obstacle', 'water', 'sky']
_C.CLASSES.COLORS = [[247, 195,  37],  # Obstacles RGB color
                     [ 41, 167, 224],  # Water RGB color
                     [ 90,  75, 164]]  # Sky RGB color

# All Paths
_C.PATHS = CN()
_C.PATHS.RESULTS       = "./results"                # Path to where the results will be saved
_C.PATHS.DATASET_ROOT  = "/path/to/dataset"         # Path to where the dataset is stored
_C.PATHS.PREDICTIONS   = "/path/to/predictions/"    # Path to where the segmentation predictions are stored


def get_cfg(config_file=None):
    cfg = _C.clone()
    if config_file is not None:
       cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

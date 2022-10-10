from PIL import Image
import numpy as np

import metrics as M


class Evaluator():

    def __init__(self):
        self.iou = M.IoU(classes=[0,1,2], class_names=['obstacle', 'water', 'sky'], ignore_idx=4)

    def evaluate_image(self, mask_pred, mask_gt, mask_inst):
        """Evaluates a single image

        Args:
            mask_pred (np.array): Predicted segmentation mask.
            mask_gt (np.array): GT segmentation mask.
            mask_inst (np.array): Instances mask.
        """
        H,W = mask_gt.shape
        # 1. Resize predicted mask
        mask_pred = np.array(Image.fromarray(mask_pred).resize((W, H), Image.NEAREST))

        # 2. Evaluate IoU
        iou_summary = self.iou.compute(mask_pred, mask_gt)

        # 3. Evaluate detection and FPs

        # 4. Evaluate water-edge
        pass


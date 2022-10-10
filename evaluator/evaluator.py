from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm

import evaluator.metrics as M

class Evaluator():

    def __init__(self, cfg):
        self.cfg = cfg

        # Read image list
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
            self.image_list = [l.strip() for l in file]

        self.iou = M.IoU(classes=cfg.CLASSES.IDS, class_names=cfg.CLASSES.NAMES, ignore_idx=cfg.CLASSES.IGNORE_ID)

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

        return {
            'mIoU': iou_summary['mIoU']
        }

    def evaluate(self, method_name):
        preds_dir = os.path.join(self.cfg.PATHS.PREDICTIONS, method_name)
        gt_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.GT_MASK_SUBDIR)
        inst_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.INST_MASK_SUBDIR)


        with tqdm(desc='Evaluating', total=len(self.image_list)) as pbar:
            for img_name in self.image_list:
                mask_pred_c = np.array(Image.open(os.path.join(preds_dir, '%s.png' % img_name)))

                # Convert color mask to class ID
                H,W,_ = mask_pred_c.shape
                mask_pred = np.full((H,W), self.cfg.CLASSES.IGNORE_ID, np.uint8)
                for cls_i, cls_c in zip(self.cfg.CLASSES.IDS, self.cfg.CLASSES.COLORS):
                    mask_cur = (mask_pred_c == np.array(cls_c)).all(2)
                    mask_pred[mask_cur] = cls_i

                mask_gt = np.array(Image.open(os.path.join(gt_dir, '%s.png' % img_name)))
                mask_inst = np.array(Image.open(os.path.join(inst_dir, '%s.png' % img_name)))

                cur_summ = self.evaluate_image(mask_pred, mask_gt, mask_inst)
                pbar.set_postfix(**cur_summ)
                pbar.update()


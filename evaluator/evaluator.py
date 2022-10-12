from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
import json

import evaluator.metrics as M

class Evaluator():

    def __init__(self, cfg):
        self.cfg = cfg

        # Read image list
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
            self.image_list = [l.strip() for l in file]

        self.iou = M.IoU(classes=cfg.CLASSES.IDS, class_names=cfg.CLASSES.NAMES, ignore_idx=cfg.CLASSES.IGNORE_ID)
        self.maritime_metrics = M.MaritimeMetrics(0, 1, 2, cfg.CLASSES.IGNORE_ID) # TODO: cfg class ids for water, ...

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
        frame_summary, overall_summary = self.iou.compute(mask_pred, mask_gt)

        # 3. Evaluate maritime metrics (WE, obst. detection and FPs)
        frame_summary_mar, overall_summary_mar = self.maritime_metrics.compute(mask_pred, mask_gt, mask_inst)

        frame_summary.update(frame_summary_mar)
        overall_summary.update(overall_summary_mar)

        return frame_summary, overall_summary

    def evaluate(self, method_name):
        preds_dir = os.path.join(self.cfg.PATHS.PREDICTIONS, method_name)
        gt_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.GT_MASK_SUBDIR)
        inst_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.INST_MASK_SUBDIR)

        frame_results = []
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

                frame_summary, overall_summary = self.evaluate_image(mask_pred, mask_gt, mask_inst)
                frame_summary['image'] = img_name
                frame_results.append(frame_summary)

                pbar.set_postfix(**overall_summary)
                pbar.update()


        # Store results (overall and per frame)
        frame_results_df = pd.DataFrame(frame_results).set_index('image')
        overall_summary = self.iou.summary()
        overall_summary.update(self.maritime_metrics.summary())

        if not osp.exists(self.cfg.PATHS.RESULTS):
            os.makedirs(self.cfg.PATHS.RESULTS)

        frame_results_df.to_csv(osp.join(self.cfg.PATHS.RESULTS, '%s_frames.csv' % method_name))
        with open(osp.join(self.cfg.PATHS.RESULTS, '%s.json' % method_name), 'w') as file:
            json.dump(overall_summary, file, indent=2)

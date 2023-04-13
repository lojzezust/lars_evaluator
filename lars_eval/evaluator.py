from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
import json

import lars_eval.context as ctx
import lars_eval.metrics as M
import lars_eval.panoptic as PM


def parse_annotations(json_data):
    # Prepare a image_name -> annotation dictionary
    id2img = {}
    for img in json_data['images']:
        id2img[img['id']] = osp.splitext(img['file_name'])[0]

    annotations = {}
    for ann in json_data['annotations']:
        img_name = id2img[ann['image_id']]
        annotations[img_name] = ann

    return annotations

class SemanticEvaluator():

    def __init__(self, cfg):
        self.cfg = cfg

        # Read image list
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
            self.image_list = [l.strip() for l in file]

        # Read panoptic annotations
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.PANOPTIC_ANN_FILE), 'r') as file:
            data = json.load(file)

        self.annotations = parse_annotations(data)

        self.iou = M.IoU(cfg)
        self.maritime_metrics = M.MaritimeMetrics(cfg)

    def evaluate_image(self, mask_pred, seg_mask, pan_mask, pan_ann, image_name):
        """Evaluates a single image

        Args:
            mask_pred (np.array): Predicted segmentation mask.
            seg_mask (np.array): GT segmentation mask.
            pan_mask (np.array): GT panoptic mask.
            pan_ann (list): GT panoptic annotations (COCO format).
            image_name (str): Name of the evaluated image (filename without extension).
        """
        H,W = seg_mask.shape
        # 1. Resize predicted mask
        mask_pred = np.array(Image.fromarray(mask_pred).resize((W, H), Image.NEAREST))

        # 2. Evaluate IoU
        frame_summary, overall_summary = self.iou.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)

        # 3. Evaluate maritime metrics (WE, obst. detection and FPs)
        frame_summary_mar, overall_summary_mar = self.maritime_metrics.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)

        frame_summary.update(frame_summary_mar)
        overall_summary.update(overall_summary_mar)

        return frame_summary, overall_summary

    def evaluate(self, preds_dir, output_dir, display_name=None):
        sem_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.SEMANTIC_MASK_SUBDIR)
        pan_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.PANOPTIC_MASK_SUBDIR)

        frame_results = []
        with tqdm(desc=display_name, total=len(self.image_list), position=ctx.PID, leave=False) as pbar:
            for img_name in self.image_list:
                mask_pred_c = np.array(Image.open(os.path.join(preds_dir, '%s.png' % img_name)))

                # Convert color mask to class ID
                H,W,_ = mask_pred_c.shape
                mask_pred = np.full((H,W), self.cfg.SEGMENTATION.IGNORE_ID, np.uint8)
                for cls_i, cls_c in zip(self.cfg.SEGMENTATION.IDS, self.cfg.SEGMENTATION.COLORS):
                    mask_cur = (mask_pred_c == np.array(cls_c)).all(2)
                    mask_pred[mask_cur] = cls_i

                mask_sem = np.array(Image.open(os.path.join(sem_dir, '%s.png' % img_name)))
                mask_pan = np.array(Image.open(os.path.join(pan_dir, '%s.png' % img_name)))
                ann_pan = self.annotations[img_name]

                frame_summary, overall_summary = self.evaluate_image(mask_pred, mask_sem, mask_pan, ann_pan, img_name)
                frame_summary['image'] = img_name
                frame_results.append(frame_summary)

                log_dict = {m:overall_summary[m] for m in self.cfg.PROGRESS.METRICS}

                pbar.set_postfix(**log_dict)
                pbar.update()


        # Store results (overall and per frame)
        frame_results_df = pd.DataFrame(frame_results).set_index('image')
        overall_summary = self.iou.summary()
        overall_summary.update(self.maritime_metrics.summary())

        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        frame_results_df.to_csv(osp.join(output_dir, 'frames.csv'))
        with open(osp.join(output_dir, 'summary.json'), 'w') as file:
            json.dump(overall_summary, file, indent=2)

        # Store extras
        self.maritime_metrics.save_extras(output_dir)

        return overall_summary


class PanopticEvaluator():

    def __init__(self, cfg):
        self.cfg = cfg

        # Read image list
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
            self.image_list = [l.strip() for l in file]

        # Read annotations
        with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.PANOPTIC_ANN_FILE), 'r') as file:
            self.ann_data = json.load(file)

            self.annotations = parse_annotations(self.ann_data)
            self.categories = {c['id']: c for c in self.ann_data['categories']}

        self.pq = PM.PQ(self.categories, cfg)
        self.pq_agnostic = PM.PQ(self.categories, cfg, class_agnostic=True, prefix='a')

        # Semantic seg metrics
        self.sem_iou = M.IoU(cfg)
        self.sem_maritime_metrics = M.MaritimeMetrics(cfg)

    def evaluate_image(self, pan_pred, pan_gt, ann_gt, sem_gt, image_name):
        """Evaluates a single image

        Args:
            pan_pred (np.array): Predicted panoptic mask.
            pan_gt (np.array): GT panoptic mask.
            ann_gt (dict): GT annotations for the image.
            image_name (str): Name of the evaluated image (filename without extension).
        """

        H,W,_ = pan_gt.shape
        # 1. Resize predicted mask
        pan_pred = np.array(Image.fromarray(pan_pred).resize((W, H), Image.NEAREST))

        # 2. Evaluate panoptic quality
        frame_summary, overall_summary = self.pq.compute(pan_pred, pan_gt, ann_gt, image_name)
        a_frame_summary, a_overall_summary = self.pq_agnostic.compute(pan_pred, pan_gt, ann_gt, image_name)

        frame_summary.update(a_frame_summary)
        overall_summary.update(a_overall_summary)

        # 3. Evaluate semantic seg metrics

        # Convert predictions to semantic seg format
        sem_pred = np.full_like(sem_gt, fill_value=self.cfg.SEGMENTATION.OBSTACLE_CLASS)
        sem_pred[pan_pred[...,0] == self.cfg.PANOPTIC.WATER_CLASS] = self.cfg.SEGMENTATION.WATER_CLASS
        sem_pred[pan_pred[...,0] == self.cfg.PANOPTIC.SKY_CLASS] = self.cfg.SEGMENTATION.SKY_CLASS

        iou_frame_summary, iou_overall_summary = self.sem_iou.compute(sem_pred, sem_gt, pan_gt, ann_gt, image_name)
        mm_frame_summary_mar, mm_overall_summary_mar = self.sem_maritime_metrics.compute(sem_pred, sem_gt, pan_gt, ann_gt, image_name)

        frame_summary.update({'sem_'+k: v for k,v in iou_frame_summary.items()})
        frame_summary.update({'sem_'+k: v for k,v in mm_frame_summary_mar.items()})

        overall_summary.update({'sem_'+k: v for k,v in iou_overall_summary.items()})
        overall_summary.update({'sem_'+k: v for k,v in mm_overall_summary_mar.items()})

        return frame_summary, overall_summary

    def evaluate(self, preds_dir, output_dir, display_name=None):
        sem_gt_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.SEMANTIC_MASK_SUBDIR)
        pan_gt_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.PANOPTIC_MASK_SUBDIR)

        frame_results = []
        with tqdm(desc=display_name, total=len(self.image_list), position=ctx.PID, leave=False) as pbar:
            for img_name in self.image_list:
                # Read panoptic predictions
                pred_pan = np.array(Image.open(os.path.join(preds_dir, '%s.png' % img_name)))

                # Read GT
                sem_gt = np.array(Image.open(os.path.join(sem_gt_dir, '%s.png' % img_name)))
                pan_gt = np.array(Image.open(os.path.join(pan_gt_dir, '%s.png' % img_name)))
                ann_gt = self.annotations[img_name]

                frame_summary, overall_summary = self.evaluate_image(pred_pan, pan_gt, ann_gt, sem_gt, img_name)
                frame_summary['image'] = img_name
                frame_results.append(frame_summary)

                log_dict = {m:overall_summary[m] for m in self.cfg.PROGRESS.METRICS}

                pbar.set_postfix(**log_dict)
                pbar.update()


        # Store results (overall and per frame)
        frame_results_df = pd.DataFrame(frame_results).set_index('image')
        overall_summary = self.pq.summary()
        overall_summary.update(self.pq_agnostic.summary())

        # Add segmentation metrics
        overall_summary.update({'sem_'+k:v for k,v in self.sem_iou.summary().items()})
        overall_summary.update({'sem_'+k:v for k,v in self.sem_maritime_metrics.summary().items()})

        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        frame_results_df.to_csv(osp.join(output_dir, 'frames.csv'))
        with open(osp.join(output_dir, 'summary.json'), 'w') as file:
            json.dump(overall_summary, file, indent=2)

        # Store extras
        self.pq.save_extras(output_dir)
        self.pq_agnostic.save_extras(output_dir, postfix='_agnostic')
        self.sem_maritime_metrics.save_extras(output_dir, postfix='_sem')

        return overall_summary

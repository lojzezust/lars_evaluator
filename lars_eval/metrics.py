

import numpy as np
import cv2
import json
import os

from lars_eval.panopticapi import rgb2id


def _get_diagonal(bbox):
    return np.sqrt(bbox[2]**2 + bbox[3]**2)

class Metric():
    def compute(self, mask_pred, mask_gt, **kwargs):
        pass

    def summary(self):
        pass

    def reset(self):
        pass

    def save_extras(self, path, **kwargs):
        pass

class IoU(Metric):
    def __init__(self, cfg):
        self.classes = cfg.SEGMENTATION.IDS
        self.class_names = cfg.SEGMENTATION.NAMES
        self.ignore_idx = cfg.SEGMENTATION.IGNORE_ID

        self.reset()

    def reset(self):
        # Metric counters
        self._total_union = {cls_i: 0 for cls_i in self.classes}
        self._total_intersection = {cls_i: 0 for cls_i in self.classes}

    def compute(self, mask_pred, gt_sem, gt_pan, ann_pan, image_name):
        frame_summary = {}
        for i,cls_i in enumerate(self.classes):
            cls_pred = (mask_pred == cls_i) & (gt_sem != self.ignore_idx)
            cls_gt = gt_sem == cls_i

            intersection = np.bitwise_and(cls_pred, cls_gt).sum()
            union = np.bitwise_or(cls_pred, cls_gt).sum()

            self._total_intersection[cls_i] += intersection
            self._total_union[cls_i] += union

            # Store current frame IoU
            cls_name = self.class_names[i] if self.class_names is not None else '%d' % cls_i
            frame_summary['IoU_%s' % cls_name] = 100. * intersection / union if union != 0 else 100.

        frame_summary['mIoU'] = sum(frame_summary.values()) / len(frame_summary)

        # Return current frame summary and overall summary
        return frame_summary, self.summary()

    def summary(self):
        results = {}
        for i, cls_i in enumerate(self.classes):
            cls_iou = 100. * self._total_intersection[cls_i] / self._total_union[cls_i]
            cls_name = self.class_names[i] if self.class_names is not None else '%d' % cls_i
            results['IoU_%s' % cls_name] = cls_iou

        results['mIoU'] = sum(results.values()) / len(results)
        return results

def dilate_mask(mask, ksize=3, it=1):
    kernel = np.ones((ksize,ksize), np.uint8)
    out = cv2.dilate(mask, kernel, iterations=it)
    return out

def erode_mask(mask, ksize=3, it=1):
    kernel = np.ones((ksize,ksize), np.uint8)
    out = cv2.erode(mask, kernel, iterations=it)
    return out

class MaritimeMetrics(Metric):
    def __init__(self, cfg):
        self.obstacle_class = cfg.SEGMENTATION.OBSTACLE_CLASS
        self.water_class = cfg.SEGMENTATION.WATER_CLASS
        self.sky_class = cfg.SEGMENTATION.SKY_CLASS
        self.ignore_idx = cfg.SEGMENTATION.IGNORE_ID

        self.cfg = cfg

        self.reset()

    def reset(self):
        # Metric counters
        self._we_total_correct = 0
        self._we_total_area = 0
        self._we_total_intersection = 0
        self._we_total_union = 0

        self._dyobs_tp = 0
        self._dyobs_fn = 0
        self._dyobs_fp = 0

        self._water_fp_area = 0
        self._water_total = 0

        # Frame data
        self._frame_data = []

    def compute(self, mask_pred, gt_sem, gt_pan, ann_pan, image_name):
        # Get water-edge area mask
        water_mask = (gt_sem == self.water_class).astype(np.uint8) # Water mask
        obstacle_mask = (gt_pan[...,0] == self.cfg.PANOPTIC.STATIC_OBSTACLE_CLASS).astype(np.uint8) # Static obstacles mask

        segment_data = []

        # 1. Dynamic obstacles recall
        valid_preds = (mask_pred == self.obstacle_class).astype(np.uint8)
        valid_preds = valid_preds & ~obstacle_mask # Remove static obstacles from predictions
        dyn_obst_anns = [ann for ann in ann_pan['segments_info'] if ann['category_id'] in self.cfg.PANOPTIC.DYN_OBST_IDS]
        gt_pan_ids = rgb2id(gt_pan)

        dyn_obst_mask_d = np.zeros_like(valid_preds)
        num_tp = 0
        num_fn = 0
        for obst_ann in dyn_obst_anns:
            obst_mask = (gt_pan_ids == obst_ann['id']).astype(np.uint8)
            obst_mask_d = dilate_mask(obst_mask, self.cfg.EVALUATION.SMALL_OBJECT_DILATION)
            dyn_obst_mask_d |= obst_mask_d
            pred_area = np.sum(valid_preds & obst_mask_d)
            total_area = np.sum(obst_mask)

            if pred_area > self.cfg.EVALUATION.MIN_COVERAGE * total_area: # TODO: IoU instead of coverage?
                self._dyobs_tp += 1
                num_tp += 1
                det_type = 'TP'
            else:
                self._dyobs_fn += 1
                num_fn += 1
                det_type = 'FN'

            # Store segment match info
            segment_data.append(dict(
                type=det_type,
                category_id=int(obst_ann['category_id']),
                coverage=pred_area/total_area,
                area=int(obst_ann['area']),
                diagonal=_get_diagonal(obst_ann['bbox'])
            ))


        # 2. Water-edge boundary accuracy
        # Get common boundary by dilation
        wo_union = obstacle_mask | water_mask
        obst_d = dilate_mask(obstacle_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        water_d = dilate_mask(water_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        we_mask = obst_d & water_d & wo_union # Get border areas
        we_mask = we_mask & ~dyn_obst_mask_d # Do not evaluate close to dynamic obstacles

        # 2.2 Update WE metric(s)
        # Accuracy inside boundary region
        we_area = we_mask.sum()
        we_correct = np.sum((gt_sem == mask_pred) * we_mask)
        self._we_total_area += we_area
        self._we_total_correct += we_correct

        # Boundary IoU (symmetric, range 0-1)
        gt_o = (gt_sem == self.obstacle_class).astype(np.uint8)
        pred_o = (mask_pred == self.obstacle_class).astype(np.uint8)
        gt_w = (gt_sem == self.water_class).astype(np.uint8)
        pred_w = (mask_pred == self.water_class).astype(np.uint8)
        i_o = np.sum((gt_o & pred_o) & we_mask)
        i_w = np.sum((gt_w & pred_w) & we_mask)
        min_i, max_i = min(i_o, i_w), max(i_o, i_w)

        we_intersection = min_i
        we_union = we_area - max_i
        self._we_total_intersection += we_intersection
        self._we_total_union += we_union


        # 3. False positive detections
        # Only evaluate inside the water regions (erode to ignore object oversegmentations)
        water_mask_e = erode_mask(water_mask, ksize=self.cfg.EVALUATION.FP_WATER_EROSION_SIZE)

        water_area = water_mask_e.sum()
        fp_mask = (mask_pred == self.obstacle_class) * water_mask_e
        conn_num, conn_mask = cv2.connectedComponents(fp_mask.astype(np.uint8))

        # Get BBoxes for FPs
        for seg_i in range(1, conn_num):
            m = conn_mask == seg_i
            ys,xs = np.where(m)
            y0,y1 = int(np.min(ys)), int(np.max(ys))
            x0,x1 = int(np.min(xs)), int(np.max(xs))

            bbox = (x0, y0, x1-x0+1, y1-y0+1)

            # Store segment match info
            segment_data.append(dict(
                type='FP',
                category_id=None,
                area=int(np.sum(m)),
                diagonal=_get_diagonal(bbox)
            ))

        num_fp = conn_num - 1
        fp_area = np.sum(fp_mask)
        self._water_total += water_area
        self._water_fp_area += fp_area
        self._dyobs_fp += num_fp

        # Metrics of the current frame
        frame_summary = {
            'WE_acc': 100. * we_correct / we_area if we_area > 0 else 100.,
            'WE_IoU': 100. * we_intersection / we_union if we_union > 0 else 100.,
            'TP': num_tp,
            'FN': num_fn,
            'FP': num_fp,
            'FPr': 100. * fp_area / water_area * 100 if water_area > 0 else 0.,
        }

        # Frame data
        self._frame_data.append(dict(
            image_name=image_name,
            segment_data=segment_data
        ))

        # Return current frame summary and overall summary
        return frame_summary, self.summary()

    def save_extras(self, path, postfix=''):
        # Save segments data
        data = {'frames': self._frame_data}
        with open(os.path.join(path, 'segments%s.json' % postfix), 'w') as file:
            json.dump(data, file, indent=2)


    def summary(self):
        pr = self._dyobs_tp / (self._dyobs_tp + self._dyobs_fp) if self._dyobs_tp + self._dyobs_fp > 0 else 0
        re = self._dyobs_tp / (self._dyobs_tp + self._dyobs_fn) if self._dyobs_tp + self._dyobs_fn > 0 else 0
        results = {
            'WE_acc': 100. * self._we_total_correct / self._we_total_area,
            'WE_IoU': 100. * self._we_total_intersection / self._we_total_union,
            'TP': self._dyobs_tp,
            'FN': self._dyobs_fn,
            'FP': self._dyobs_fp,
            'FPr': self._water_fp_area / self._water_total * 100.,
            'Pr': 100. * pr,
            'Re': 100. * re,
            'F1': 100. * 2 * pr * re / (pr + re) if pr + re > 0 else 0
        }

        return results

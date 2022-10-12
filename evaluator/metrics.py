

import numpy as np
import cv2

class Metric():
    def compute(self, mask_pred, mask_gt, **kwargs):
        pass

    def summary(self):
        pass

    def reset(self):
        pass

class IoU(Metric):
    def __init__(self, cfg):
        self.classes = cfg.CLASSES.IDS
        self.class_names = cfg.CLASSES.NAMES
        self.ignore_idx = cfg.CLASSES.IGNORE_ID

        self.reset()

    def reset(self):
        # Metric counters
        self._total_union = {cls_i: 0 for cls_i in self.classes}
        self._total_intersection = {cls_i: 0 for cls_i in self.classes}

    def compute(self, mask_pred, mask_gt):
        frame_summary = {}
        for i,cls_i in enumerate(self.classes):
            cls_pred = (mask_pred == cls_i) & (mask_gt != self.ignore_idx)
            cls_gt = mask_gt == cls_i

            intersection = np.bitwise_and(cls_pred, cls_gt).sum()
            union = np.bitwise_or(cls_pred, cls_gt).sum()

            self._total_intersection[cls_i] += intersection
            self._total_union[cls_i] += union

            # Store current frame IoU
            cls_name = self.class_names[i] if self.class_names is not None else '%d' % cls_i
            frame_summary['IoU_%s' % cls_name] = intersection / union if union != 0 else 1.

        frame_summary['mIoU'] = sum(frame_summary.values()) / len(frame_summary)

        # Return current frame summary and overall summary
        return frame_summary, self.summary()

    def summary(self):
        results = {}
        for i, cls_i in enumerate(self.classes):
            cls_iou = self._total_intersection[cls_i] / self._total_union[cls_i]
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
        self.obstacle_class = cfg.CLASSES.OBSTACLE_CLASS
        self.water_class = cfg.CLASSES.WATER_CLASS
        self.sky_class = cfg.CLASSES.SKY_CLASS
        self.ignore_idx = cfg.CLASSES.IGNORE_ID

        self.cfg = cfg

        self.reset()

    def reset(self):
        # Metric counters
        self._we_total_correct = 0
        self._we_total_area = 0

        self._dyobs_tp = 0
        self._dyobs_fn = 0
        self._dyobs_fp = 0

        self._water_fp_area = 0
        self._water_total = 0

    def compute(self, mask_pred, mask_gt, mask_inst):
        # 1.1 Get water-edge area mask
        water_mask = (mask_gt == self.water_class).astype(np.uint8)
        obstacle_mask = (mask_gt == self.obstacle_class).astype(np.uint8) # TODO: only static obstacles
        obstacle_mask = obstacle_mask & ~(mask_inst > 0)

        obst_d = dilate_mask(obstacle_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        water_d = dilate_mask(water_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        we_mask = obst_d & water_d # TODO: ignore regions

        # 1.2 Update WE metric(s)
        we_area = we_mask.sum()
        we_correct = np.sum((mask_gt == mask_pred) * we_mask)
        self._we_total_area += we_area
        self._we_total_correct += we_correct

        # 2. Dynamic obstacles recall
        valid_preds = (mask_pred == self.obstacle_class).astype(np.uint8)
        valid_preds = valid_preds & ~obstacle_mask # Remove static obstacles from predictions
        dyn_obst_mask_d = np.zeros_like(valid_preds)

        num_tp = 0
        num_fn = 0
        for obst_i in np.unique(mask_inst):
            if obst_i==0: continue

            obst_mask = (mask_inst == obst_i).astype(np.uint8)
            obst_mask_d = dilate_mask(obst_mask, self.cfg.EVALUATION.SMALL_OBJECT_DILATION)
            dyn_obst_mask_d |= obst_mask_d
            pred_area = np.sum(valid_preds & obst_mask_d)
            total_area = np.sum(obst_mask)

            if pred_area > self.cfg.EVALUATION.MIN_COVERAGE * total_area:
                self._dyobs_tp += 1
                num_tp += 1
            else:
                self._dyobs_fn += 1
                num_fn += 1


        # 3. False positive detections
        # Only evaluate inside the water regions (erode to ignore object oversegmentations)
        water_mask_e = erode_mask(water_mask, ksize=self.cfg.EVALUATION.FP_WATER_EROSION_SIZE)

        water_area = water_mask_e.sum()
        fp_mask = (mask_pred == self.obstacle_class) * water_mask_e
        num_fp, _ = cv2.connectedComponents(fp_mask.astype(np.uint8))
        num_fp -= 1
        fp_area = np.sum(fp_mask)
        self._water_total += water_area
        self._water_fp_area += fp_area
        self._dyobs_fp += num_fp

        # Metrics of the current frame
        frame_summary = {
            'WE_acc': we_correct / we_area if we_area > 0 else 1.,
            'TP': num_tp,
            'FN': num_fn,
            'FP': num_fp,
            'FPr': fp_area / water_area * 100 if water_area > 0 else 0.,
        }

        # Return current frame summary and overall summary
        return frame_summary, self.summary()

    def summary(self):
        pr = self._dyobs_tp / (self._dyobs_tp + self._dyobs_fp)
        re = self._dyobs_tp / (self._dyobs_tp + self._dyobs_fn)
        results = {
            'WE_acc': self._we_total_correct / self._we_total_area,
            'TP': self._dyobs_tp,
            'FN': self._dyobs_fn,
            'FP': self._dyobs_fp,
            'FPr': self._water_fp_area / self._water_total * 100.,
            'Pr': pr,
            'Re': re,
            'F1': 2 * pr * re / (pr + re)
        }

        return results

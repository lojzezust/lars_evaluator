

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
    def __init__(self, classes, class_names=None, ignore_idx=4):
        self.classes = classes
        self.class_names = class_names
        self.ignore_idx = ignore_idx

        self.reset()

    def reset(self):
        # Metric counters
        self._total_union = {cls_i: 0 for cls_i in self.classes}
        self._total_intersection = {cls_i: 0 for cls_i in self.classes}

    def compute(self, mask_pred, mask_gt):
        for cls_i in self.classes:
            cls_pred = (mask_pred == cls_i) & (mask_gt != self.ignore_idx)
            cls_gt = mask_gt == cls_i

            intersection = np.bitwise_and(cls_pred, cls_gt).sum()
            union = np.bitwise_or(cls_pred, cls_gt).sum()

            self._total_intersection[cls_i] += intersection
            self._total_union[cls_i] += union

        # Return current summary
        return self.summary()

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

class MaritimeMetrics(Metric):
    def __init__(self, obstacle_class=0, water_class=1, sky_class=2, ignore_idx=4):
        self.obstacle_class = obstacle_class
        self.water_class = water_class
        self.sky_class = sky_class
        self.ignore_idx = ignore_idx

        self.reset()

    def reset(self):
        # Metric counters
        self._we_total_correct = 0
        self._we_total_area = 0

    def compute(self, mask_pred, mask_gt):
        # 1.1 Get water-edge area mask
        water_mask = (mask_gt == self.water_class).astype(np.uint8)
        obstacle_mask = (mask_gt == self.obstacle_class).astype(np.uint8) # TODO: only static obstacles

        # TODO: configurable dilation width
        obst_d = dilate_mask(obstacle_mask, 11)
        water_d = dilate_mask(water_mask, 11)
        we_mask = obst_d & water_d # TODO: ignore regions

        # 1.2 Update WE metric(s)
        self._we_total_area += we_mask.sum()
        self._we_total_correct += np.sum((mask_gt == mask_pred) * we_mask)

        # TODO: Dynamic obstacles and FP detections

        # Return current summary
        return self.summary()

    def summary(self):
        results = {
            'WE_acc': self._we_total_correct / self._we_total_area
        }

        return results

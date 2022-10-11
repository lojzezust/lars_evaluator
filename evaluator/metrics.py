

import numpy as np

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

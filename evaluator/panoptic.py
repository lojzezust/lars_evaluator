"""Panoptic Quality (PQ) metric adapted from MMDet and `panopticapi` (TODO)"""
import os
import numpy as np
from panopticapi.evaluation import OFFSET, VOID, PQStat
from panopticapi.utils import rgb2id

from evaluator.metrics import Metric


class PanopticMetric(Metric):
    def compute(self, pan_pred, pan_gt, ann_gt, **kwargs):
        pass


def parse_pq_results(pq_results):
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_th'] = 100 * pq_results['Things']['pq']
    result['SQ_th'] = 100 * pq_results['Things']['sq']
    result['RQ_th'] = 100 * pq_results['Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']

    return result

class PQ(PanopticMetric):
    def __init__(self, categories, cfg):
        # TODO: cfg for void, etc.
        self.categories = categories


        self._pq_stat = PQStat()
        self._pq_stat_frame = None

    def compute(self, pan_pred, pan_gt, ann_gt, **kwargs):
        self._pq_stat_frame = PQStat()

        # Convert predictions into individual components
        pan_pred_id = rgb2id(pan_pred) # Segment IDs
        pan_pred_cls = pan_pred[..., 0] # Class IDs
        pan_pred_inst = pan_pred[..., 1] * 256 + pan_pred[..., 2] # Instance IDs

        labels, labels_cnt = np.unique(pan_pred_id, return_counts=True)

        # Generate segment info data
        ann_pred = {}
        for label, label_cnt in zip(labels, labels_cnt):
            # TODO: void! Ignore regions are treated as void?
            cat_preds = pan_pred_cls[pan_pred_id == label]

            ann_pred[label] = {
                'id': label,
                'area': label_cnt,
                'category_id': np.unique(cat_preds)[0], # Most common predicted class of the segment
            }

            # TODO: check if category_id is valid

        # Convert GT
        pan_gt_id = rgb2id(pan_gt)
        gt_segms = {el['id']: el for el in ann_gt['segments_info']}
        pred_segms = {el['id']: el for el in ann_pred}

        # Find segment matches
        pan_gt_pred = pan_gt_id.astype(np.uint64) * OFFSET + pan_pred_id.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs (true positives)
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                    'category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                'area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self._pq_stat_frame[gt_segms[gt_label]['category_id']].tp += 1
                self._pq_stat_frame[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored TODO: iscrowd annotations are different?
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            self._pq_stat_frame[gt_info['category_id']].fn += 1


        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info['category_id']], pred_label),
                    0)
            # predicted segment is ignored if more than half of
            # the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            self._pq_stat_frame[pred_info['category_id']].fp += 1

        # Update global count
        self._pq_stat += self._pq_stat_frame


    def summary(self):
        frame_summary = self._get_summary(self._pq_stat_frame)
        overall_summary = self._get_summary(self._pq_stat)

        return frame_summary, overall_summary

    def _get_summary(self, pq_stat):
        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        for name, isthing in metrics:
            pq_results[name], classwise_results = self._pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        # TODO: classwise results?

        result = dict()
        result['PQ'] = 100 * pq_results['All']['pq']
        result['SQ'] = 100 * pq_results['All']['sq']
        result['RQ'] = 100 * pq_results['All']['rq']
        result['PQ_th'] = 100 * pq_results['Things']['pq']
        result['SQ_th'] = 100 * pq_results['Things']['sq']
        result['RQ_th'] = 100 * pq_results['Things']['rq']
        result['PQ_st'] = 100 * pq_results['Stuff']['pq']
        result['SQ_st'] = 100 * pq_results['Stuff']['sq']
        result['RQ_st'] = 100 * pq_results['Stuff']['rq']

        return result

    def reset(self):
        self._pq_stat = PQStat()

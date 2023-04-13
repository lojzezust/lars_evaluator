"""Panoptic Quality (PQ) metric adapted from MMDet and `panopticapi` (TODO)"""
import os
import numpy as np
from panopticapi.evaluation import OFFSET, VOID, PQStat
from panopticapi.utils import rgb2id
import pandas as pd
import json

from lars_eval.metrics import Metric, dilate_mask


class PanopticMetric(Metric):
    def compute(self, pan_pred, pan_gt, ann_gt, **kwargs):
        pass


def _get_bbox(mask):
    """Computes the bounds of the mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    bbox = int(xmin), int(ymin), int(xmax-xmin) + 1, int(ymax-ymin) + 1
    return bbox

class PQ(PanopticMetric):
    def __init__(self, categories, cfg, class_agnostic=False, prefix=''):
        self.categories = categories
        self.cfg = cfg
        self.class_agnostic = class_agnostic
        self.prefix = prefix

        if self.class_agnostic:
            self.agnostic_id = 20 # TODO: config
            self.class_agnostic_cat = {cid: cat for cid, cat in categories.items() if cat['isthing'] == 0}
            self.class_agnostic_cat[self.agnostic_id] = {
                'id': self.agnostic_id,
                'name': 'Dynamic Obstacle',
                'supercategory': 'obstacle',
                'isthing': 1,
                'color': [255, 39, 43]
            }

        # Init metrics
        self.reset()

    def _resolve_id(self, cat_id):
        """Resolve category id (if agnostic group all thing IDs)."""
        isthing = self.categories[cat_id]['isthing']

        if self.class_agnostic and isthing:
            cat_id = self.agnostic_id

        return cat_id

    def compute(self, pan_pred, pan_gt, ann_gt, image_name, **kwargs):
        # 1. Water-edge boundary accuracy
        # Get common boundary by dilation
        water_mask = (pan_gt[...,0] == self.cfg.PANOPTIC.WATER_CLASS).astype(np.uint8)
        obstacle_mask = (pan_gt[...,0] == self.cfg.PANOPTIC.STATIC_OBSTACLE_CLASS).astype(np.uint8)
        dyn_obst_mask = (pan_gt[...,0] >= 10).astype(np.uint8)

        wo_union = obstacle_mask | water_mask
        obst_d = dilate_mask(obstacle_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        water_d = dilate_mask(water_mask, self.cfg.EVALUATION.WE_DILATION_SIZE)
        dyn_obst_mask_d = dilate_mask(dyn_obst_mask, self.cfg.EVALUATION.SMALL_OBJECT_DILATION)

        we_mask = obst_d & water_d & wo_union # Get border areas
        we_mask = we_mask & ~dyn_obst_mask_d # Do not evaluate close to dynamic obstacles

        # 1.2 Update WE metric(s)
        # Accuracy inside boundary region
        we_area = we_mask.sum()
        we_correct = np.sum((pan_gt[...,0] == pan_pred[...,0]) * we_mask)
        self._we_total_area += we_area
        self._we_total_correct += we_correct

        # 2. PQ metrics
        self._pq_stat_frame = PQStat()

        categories = self.categories
        if self.class_agnostic:
            categories = self.class_agnostic_cat

        # Convert predictions into individual components
        pan_pred_id = rgb2id(pan_pred) # Segment IDs
        pan_pred_cls = pan_pred[..., 0] # Class IDs
        pan_pred_inst = pan_pred[..., 1] * 256 + pan_pred[..., 2] # Instance IDs

        labels, labels_cnt = np.unique(pan_pred_id, return_counts=True)

        # Generate segment info data
        ann_pred = []
        for label, label_cnt in zip(labels, labels_cnt):
            cat_preds = pan_pred_cls[pan_pred_id == label]
            cat_id = np.unique(cat_preds)[0] # Most common predicted class of the segment
            # Ignore void regions
            if cat_id == self.cfg.PANOPTIC.VOID_ID:
                continue

            ann_pred.append({
                'id': label,
                'area': label_cnt,
                'category_id': cat_id,
                'bbox': _get_bbox(pan_pred_id == label)
            })

            # TODO: check if category_id is valid


        # Convert GT
        pan_gt_id = rgb2id(pan_gt)
        gt_segms = {el['id']: el for el in ann_gt['segments_info']}
        pred_segms = {el['id']: el for el in ann_pred}
        gt_segm_matches = {}

        # Find segment matches
        pan_gt_pred = pan_gt_id.astype(np.uint64) * OFFSET + pan_pred_id.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

            # For each GT segment store segment with max intersection
            if gt_id not in gt_segm_matches:
                gt_segm_matches[gt_id] = (0, None)

            if intersection > gt_segm_matches[gt_id][0]:
                gt_segm_matches[gt_id] = (intersection, pred_id)


        # Store matched segment category ids (for confusion_matrix)
        for gt_id in gt_segm_matches:
            intersection, pred_id = gt_segm_matches[gt_id]

            if gt_id not in gt_segms:
                continue
            gt_cat_id = self._resolve_id(gt_segms[gt_id]['category_id'])

            if pred_id == 0.0:
                self._matched_segments.append((-1, gt_cat_id, None))
                continue

            pred_cat_id = self._resolve_id(pred_segms[pred_id]['category_id'])

            union = pred_segms[pred_id]['area'] + gt_segms[gt_id][
                'area'] - intersection - gt_pred_map.get((VOID, pred_id), 0)
            iou = intersection / union


            self._matched_segments.append((pred_cat_id, gt_cat_id, iou))


        # count all matched pairs (true positives)
        segment_data = []
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

            gt_cat_id = self._resolve_id(gt_segms[gt_label]['category_id'])
            pred_cat_id = self._resolve_id(pred_segms[pred_label]['category_id'])

            if gt_cat_id != pred_cat_id:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                'area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self._pq_stat_frame[gt_cat_id].tp += 1
                self._pq_stat_frame[gt_cat_id].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

                # Store segment match info
                segment_data.append(dict(
                    type='TP',
                    gt_label=int(gt_label),
                    pred_label=int(pred_label),
                    iou=iou,
                    category_id=int(gt_cat_id),
                    gt_area=int(gt_segms[gt_label]['area']),
                    pred_area=int(pred_segms[pred_label]['area']),
                    gt_bbox=gt_segms[gt_label]['bbox'],
                    pred_bbox=pred_segms[pred_label]['bbox']
                ))

        # count false negatives
        crowd_labels_dict = {}
        static_obst_lbl = None
        for gt_label, gt_info in gt_segms.items():
            cat_id = self._resolve_id(gt_info['category_id'])
            # Store static obstacle panoptic label
            if cat_id == self.cfg.PANOPTIC.STATIC_OBSTACLE_CLASS:
                static_obst_lbl = gt_label

            if gt_label in gt_matched:
                continue

            # Ignore crowd segments for FNs and store their ids
            if gt_info['iscrowd'] == 1:
                # TODO: still count crowd segments as FN if classified as water?
                crowd_labels_dict[cat_id] = gt_label

                # If less than half of segment is covered non-obstacle predictions (void, water, sky), don't count as FN
                intersection = 0
                for seg_id in [self.cfg.PANOPTIC.VOID_ID, self.cfg.PANOPTIC.WATER_CLASS, self.cfg.PANOPTIC.SKY_CLASS]:
                    intersection += gt_pred_map.get((seg_id, pred_label), 0)

                if intersection / gt_info['area'] < 0.5:
                    continue

            self._pq_stat_frame[cat_id].fn += 1

            # Store segment match info
            segment_data.append(dict(
                type='FN',
                gt_label=int(gt_label),
                category_id=int(cat_id),
                gt_area=int(gt_info['area']),
                gt_bbox=gt_info['bbox'],
            ))

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue

            # intersection of the segment with VOID
            intersection = gt_pred_map.get((self.cfg.PANOPTIC.VOID_ID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            cat_id = self._resolve_id(pred_info['category_id'])
            if cat_id in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[cat_id], pred_label), 0)

            # if thing: plus intersection with static obstacle region
            if categories[cat_id]['isthing'] == 1:
                intersection += gt_pred_map.get(
                    (static_obst_lbl, pred_label), 0)

            # predicted segment is ignored if more than half of
            # the segment correspond to VOID and CROWD regions
            # Predicted obstacles are also ignored if more than half of
            # the segment corresponds to STATIC OBSTACLES
            if intersection / pred_info['area'] > 0.5:
                continue

            self._pq_stat_frame[cat_id].fp += 1

            # Store segment match info
            segment_data.append(dict(
                type='FP',
                pred_label=int(pred_label),
                category_id=int(cat_id),
                pred_area=int(pred_info['area']),
                pred_bbox=pred_info['bbox']
            ))

        # Update global count
        self._pq_stat += self._pq_stat_frame

        frame_summary = self._get_summary(self._pq_stat_frame)
        frame_summary['WE_acc'] = 100. * we_correct / we_area if we_area > 0 else 100.
        overall_summary = self.summary()

        # Frame data
        self._frame_data.append(dict(
            image_name=image_name,
            segment_data=segment_data,
            we_accuracy=float(frame_summary['WE_acc'])
        ))

        return frame_summary, overall_summary


    def summary(self):
        overall_summary = self._get_summary(self._pq_stat)
        overall_summary['WE_acc'] = 100. * self._we_total_correct / self._we_total_area

        return overall_summary

    def _get_summary(self, pq_stat):
        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        categories = self.categories
        if self.class_agnostic:
            categories = self.class_agnostic_cat

        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        result = dict()
        result[self.prefix + 'PQ'] = 100. * pq_results['All']['pq']
        result[self.prefix + 'SQ'] = 100. * pq_results['All']['sq']
        result[self.prefix + 'RQ'] = 100. * pq_results['All']['rq']
        result[self.prefix + 'PQ_th'] = 100. * pq_results['Things']['pq']
        result[self.prefix + 'SQ_th'] = 100. * pq_results['Things']['sq']
        result[self.prefix + 'RQ_th'] = 100. * pq_results['Things']['rq']
        result[self.prefix + 'PQ_st'] = 100. * pq_results['Stuff']['pq']
        result[self.prefix + 'SQ_st'] = 100. * pq_results['Stuff']['sq']
        result[self.prefix + 'RQ_st'] = 100. * pq_results['Stuff']['rq']

        return result

    def save_extras(self, path, postfix=''):
        # Save matched category pairs
        df = pd.DataFrame(self._matched_segments, columns=['pred', 'gt', 'iou'])
        df.to_csv(os.path.join(path, 'obst_cls%s.csv' % postfix), index=False)

        # Save segments data
        data = {'frames': self._frame_data}
        with open(os.path.join(path, 'segments%s.json' % postfix), 'w') as file:
            json.dump(data, file, indent=2)

    def reset(self):
        self._pq_stat = PQStat()
        self._pq_stat_frame = PQStat()
        self._matched_segments = []
        self._frame_data = []

        # WE metrics
        self._we_total_area = 0
        self._we_total_correct = 0

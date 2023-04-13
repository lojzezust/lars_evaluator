import json
import pandas as pd
import numpy as np
import os
import os.path as osp
import argparse

import matplotlib as mpl
from tqdm.auto import tqdm
from PIL import Image
from lars_eval.panopticapi import rgb2id, id2rgb
import cv2
import seaborn as sns

SEGMENTS_DIR = '/home/lojze/development/lars_evaluator/results_panoptic'
SEGMENTS_POSTFIX = '_segments_agnostic.json'
PANOPTIC_ANN = '/home/lojze/data/datasets/LaRS/split/test/panoptic_annotations.json'
OUTPUT_DIR = '/home/lojze/data/examples/LaRS_pan_det'
IMAGES_DIR = '/home/lojze/data/datasets/LaRS/split/test/images'
PRED_MASK_DIR = '/home/lojze/data/predictions/lars/panoptic'
GT_MASK_DIR = '/home/lojze/data/datasets/LaRS/split/test/panoptic_masks'

PADDING = 64
NUM_COLORS = 6
NUM_COLOR_VARIATIONS = 3

ALPHA = 0.2
CAT_COLORS = {
    0: [0,0,0],
    1: [255,212,25],
    3: [70,245,255],
    5: [170,0,255]
}

def get_args(args=None):
    parser = argparse.ArgumentParser(description='Visualizes panoptic detections (TPs, FPs, FNs)')
    parser.add_argument('method', type=str, help='Method to visualize')
    parser.add_argument('--output_dir', type=str, help='Root output directory', default=OUTPUT_DIR)
    parser.add_argument('--images_dir', type=str, help='Source of images', default=IMAGES_DIR)
    parser.add_argument('--predictions_dir', type=str, help='Root of predictions', default=PRED_MASK_DIR)
    parser.add_argument('--gt_dir', type=str, help='GT masks dir', default=GT_MASK_DIR)
    parser.add_argument('--gt_ann', type=str, help='Source of GT annotations', default=PANOPTIC_ANN)
    parser.add_argument('--segments_dir', type=str, help='Directory where the segments files are located', default=SEGMENTS_DIR)
    parser.add_argument('--segments_postfix', type=str, help='Filename postfix for segments file', default=SEGMENTS_POSTFIX)

    return parser.parse_args(args)

def paint_mask(img, mask, color, opacity=1.0):
    mask = mask[..., None]
    cmask = mask * np.array(color)[None,None]
    cmask_o = (opacity * cmask + (1-opacity) * img)
    img_n = cmask_o * mask + img * (1-mask)

    return img_n.astype(np.uint8)

def main():
    args = get_args()

    with open(osp.join(args.segments_dir, args.method + args.segments_postfix), 'r') as file:
        data = json.load(file)

    with open(args.gt_ann, 'r') as file:
        ann_data = json.load(file)

    # Generate color palette
    colors = sns.color_palette('hls', desat=1, n_colors=NUM_COLORS * NUM_COLOR_VARIATIONS)
    colors = np.array(colors).reshape((NUM_COLOR_VARIATIONS,NUM_COLORS,3))
    colors = colors.transpose(1,0,2).reshape((NUM_COLORS * NUM_COLOR_VARIATIONS,3))[:-1]
    colors = (colors * 255).astype(np.uint8)

    # Read segment data (evaluator)
    det_data = []
    for frame in data['frames']:
        for seg in frame['segment_data']:
            d = {'image': frame['image_name']}
            d.update(seg)

            det_data.append(d)

    # Prepare a image_name -> annotation dictionary
    id2img = {}
    for img in ann_data['images']:
        id2img[img['id']] = osp.splitext(img['file_name'])[0]

    # Read annotations
    for ann in ann_data['annotations']:
        img_name = id2img[ann['image_id']]
        for seg in ann['segments_info']:
            d = {
                'image': img_name,
                'type': 'GT',
                'gt_label': seg['id'],
                'gt_bbox': seg['bbox'],
                'gt_area': seg['area'],
                'category_id': seg['category_id']
            }

            det_data.append(d)

    det_df = pd.DataFrame(det_data)

    output_dir = osp.join(args.output_dir, args.method)
    pred_mask_dir = osp.join(args.predictions_dir, args.method)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    im_gr = det_df.groupby('image')

    for img_name, img_df in tqdm(im_gr, desc='Exporting examples'):
        img_path = osp.join(args.images_dir, img_name + '.jpg')
        if not osp.exists(img_path):
            print('[WARN] Missing image: %s' % img_name)
            continue

        img = np.array(Image.open(img_path))
        pred_mask = np.array(Image.open(osp.join(pred_mask_dir, img_name + '.png')))
        gt_mask = np.array(Image.open(osp.join(args.gt_dir, img_name + '.png')))
        pred_id_mask = rgb2id(pred_mask)
        gt_id_mask = rgb2id(gt_mask)


        vis_images = []
        for det_type in ['GT', 'PRED', 'TP', 'FP', 'FN']:
            if det_type == 'PRED' or det_type == 'GT':
                img_cur = np.copy(img)
                cur_mask = pred_id_mask if det_type=='PRED' else gt_id_mask
                i = 0
                for v in np.unique(cur_mask):
                    cat_id,_,_ = id2rgb(v)
                    if cat_id in CAT_COLORS:
                        color = CAT_COLORS[cat_id]
                        img_cur[cur_mask == v] = color
                    else:
                        color = colors[i % len(colors)]
                        i+=1

                        mask = cur_mask == v
                        img_cur = paint_mask(img_cur, mask, color, opacity=ALPHA)

                        obst_m = mask.astype(np.uint8)
                        obst_b = cv2.dilate(obst_m, np.ones((9,9))) - obst_m
                        img_cur = paint_mask(img_cur, obst_b, color, opacity=1)

                img_cur = (img * 0.3 + img_cur * 0.7).astype(np.uint8)
            else:
                img_cur = np.copy(img)
                img_df_t = img_df.loc[img_df['type'] == det_type]
                for i,(ix,sample) in enumerate(img_df_t.iterrows()):
                    if det_type in ['TP', 'GT'] and sample['category_id'] < 10:
                        continue

                    cur_mask = pred_id_mask if det_type in ['TP','FP'] else gt_id_mask
                    cur_label = sample['pred_label'] if det_type in ['TP','FP'] else sample['gt_label']

                    color = colors[i % len(colors)]
                    mask = cur_mask==cur_label
                    img_cur = paint_mask(img_cur, mask, color, opacity=ALPHA)

                    # Add border
                    obst_m = mask.astype(np.uint8)
                    obst_b = cv2.dilate(obst_m, np.ones((9,9))) - obst_m
                    img_cur = paint_mask(img_cur, obst_b, color, opacity=1)

            cv2.putText(img_cur, det_type, (img_cur.shape[1]//2, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (100,100,100), 6, cv2.LINE_AA)
            vis_images.append(img_cur)

        row1 = np.hstack((vis_images[0], vis_images[1]))
        row2 = np.hstack((vis_images[2], vis_images[3], vis_images[4]))
        w = row1.shape[1]
        h = int(row2.shape[0] / row2.shape[1] * w)
        row2 = cv2.resize(row2, (w,h))
        final_img = np.vstack((row1,row2))

        # Save image
        output_path = osp.join(output_dir, '%s.jpg' % img_name)
        Image.fromarray(final_img).save(output_path)

if __name__=='__main__':
    main()

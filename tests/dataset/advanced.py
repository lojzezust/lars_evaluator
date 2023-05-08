"""Simple tests to verify the corectness of the LaRS dataset."""

import unittest
import os
import os.path as osp
from PIL import Image
import filecmp
import random
import numpy as np
from tqdm.auto import tqdm

DATASET_ROOT = osp.expanduser('~/data/datasets/LaRS/split_v0.9.3')
# DATASET_ROOT = osp.expanduser('/storage/datasets/marine/LaRS_v0.9.1')

THRESHOLD_HI = 1
THRESHOLD_LO = 0.001

def img_diff(img1, img2, threshold=2):
    diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))

    return diff

class TestLaRSAdvanced(unittest.TestCase):

    def setUp(self):
        self.sets = {}

        for ds_set in ['train', 'test', 'val']:
            with open(osp.join(DATASET_ROOT, ds_set, 'image_list.txt'), 'r') as file:
                self.sets[ds_set] = [l.strip() for l in file]

    def test_seq_match(self):
        """Test that the files with the same name in the images_seq and images folders match."""

        SUBSET_SIZE = None

        for ds_set in self.sets:
            imgs = self.sets[ds_set]
            if SUBSET_SIZE is not None:
                # Test on random subset of images.
                random.shuffle(imgs)
                imgs = imgs[:SUBSET_SIZE]

            with self.subTest(set=ds_set):
                for name in imgs:
                    img = osp.join(DATASET_ROOT, ds_set, 'images', '%s.jpg' % name)
                    img_seq = osp.join(DATASET_ROOT, ds_set, 'images_seq', '%s.jpg' % name)
                    # Check files match with filecmp
                    self.assertTrue(filecmp.cmp(img, img_seq), "Files do not match: %s" % name)

    def test_no_duplicates(self):
        """Test that there are no duplicate sequential frames in the dataset."""

        for ds_set in self.sets:
            with self.subTest(set=ds_set):
                seq_dir = osp.join(DATASET_ROOT, ds_set, 'images_seq')

                images = sorted(os.listdir(seq_dir))

                prev_img_name = osp.splitext(images[0])[0]
                prev_img = np.array(Image.open(osp.join(seq_dir, images[0])))
                for img_fn in tqdm(images[1:], desc=ds_set):
                    img_name = osp.splitext(img_fn)[0]
                    img = np.array(Image.open(osp.join(seq_dir, img_fn)))

                    prev_t = prev_img_name.split('_')
                    prev_seq = "_".join(prev_t[:-1])
                    prev_f = int(prev_t[-1])
                    cur_t = img_name.split('_')
                    cur_seq = "_".join(cur_t[:-1])
                    cur_f = int(cur_t[-1])

                    if (prev_seq == cur_seq) and abs(prev_f - cur_f)==1:
                        diff = img_diff(prev_img, img)
                        if diff < THRESHOLD_HI:
                            tqdm.write("WARN: Potential duplicate: %s - %d & %d" % (cur_seq, prev_f, cur_f))
                        self.assertTrue(diff > THRESHOLD_LO, "Duplicate frames %d & %d for sequence %s" % (prev_f, cur_f, cur_seq))

                    prev_img_name = img_name
                    prev_img = img


if __name__ == '__main__':
    unittest.main()

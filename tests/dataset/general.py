"""Simple tests to verify the corectness of the LaRS dataset."""

import unittest
import os
import os.path as osp
from PIL import Image

DATASET_ROOT = osp.expanduser('~/data/datasets/LaRS/split')

class TestLaRSGeneral(unittest.TestCase):

    def setUp(self):
        self.sets = {}

        for ds_set in ['train', 'test', 'val', 'all']:
            with open(osp.join(DATASET_ROOT, ds_set, 'image_list.txt'), 'r') as file:
                self.sets[ds_set] = [l.strip() for l in file]

    def test_overlap(self):
        """Test that the training, validation and test sets do not overlap."""

        # TODO: implement
        pass

    def test_exists(self):
        """Tests that all the images, masks and instance_masks exist."""

        for ds_set in self.sets:
            imgs = self.sets[ds_set]

            with self.subTest(set=ds_set):
                self.assertTrue(osp.exists(osp.join(DATASET_ROOT, ds_set, 'panoptic_annotations.json')))
                for name in imgs:
                    img = osp.join(DATASET_ROOT, ds_set, 'images' , '%s.jpg' % name)
                    pan_mask = osp.join(DATASET_ROOT, ds_set, 'panoptic_masks' , '%s.png' % name)
                    sem_mask = osp.join(DATASET_ROOT, ds_set, 'semantic_masks' , '%s.png' % name)

                    self.assertTrue(osp.exists(img), 'Missing image: %s' % name)
                    self.assertTrue(osp.exists(pan_mask), 'Missing mask: %s' % name)
                    self.assertTrue(osp.exists(sem_mask), 'Missing instances: %s' % name)


    def test_size_match(self):
        """Tests if the sizes of images and masks match."""

        for ds_set in self.sets:
            imgs = self.sets[ds_set]

            with self.subTest(set=ds_set):
                for name in imgs:
                    img = Image.open(osp.join(DATASET_ROOT, ds_set, 'images', '%s.jpg' % name))
                    pan_mask = Image.open(osp.join(DATASET_ROOT, ds_set, 'panoptic_masks', '%s.png' % name))
                    sem_mask = Image.open(osp.join(DATASET_ROOT, ds_set, 'semantic_masks', '%s.png' % name))
                    self.assertEqual(img.size, pan_mask.size, msg='Size mismatch: %s' % name)
                    self.assertEqual(img.size, sem_mask.size, msg='Size mismatch: %s' % name)
                    img.close()
                    pan_mask.close()
                    sem_mask.close()

    def test_seq(self):
        """Test that all images contain history data (at least 9 previous frames)."""

        for ds_set in self.sets:
            imgs = self.sets[ds_set]

            with self.subTest(set=ds_set):
                for name in imgs:
                    tokens = name.split('_')
                    seq_name, frame_i = '_'.join(tokens[:-1]), int(tokens[-1])
                    for i in range(10):
                        frame_cur = frame_i - i
                        frame_fn = '%s_%05d.jpg' % (seq_name, frame_cur)
                        img = osp.join(DATASET_ROOT, ds_set, 'images_seq', frame_fn)
                        self.assertTrue(osp.exists(img), "Missing frame: %s" % frame_fn)

    def test_panoptic(self):
        """Test panoptic masks."""

        # TODO: implement


        pass


if __name__ == '__main__':
    unittest.main()

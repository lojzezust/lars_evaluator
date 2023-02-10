"""Simple tests to verify the corectness of the LaRS dataset."""

import unittest
import os
import os.path as osp
from PIL import Image
import json

DATASET_ROOT = osp.expanduser('~/data/datasets/LaRS/split')

class TestLaRSAnnotations(unittest.TestCase):

    def setUp(self):
        self.sets = {}

        for ds_set in ['train', 'test', 'val', 'all']:
            with open(osp.join(DATASET_ROOT, ds_set, 'panoptic_annotations.json'), 'r') as file:
                self.sets[ds_set] = json.load(file)

    def test_unique(self):
        """Test that each image contains a single annotation entry."""

        for ds_set in self.sets:
            data = self.sets[ds_set]

            done = set()
            with self.subTest(set=ds_set):
                for ann in data['annotations']:
                    self.assertTrue(ann['image_id'] not in done, ann['file_name'])
                    done.add(ann['image_id'])

    def test_iscrowd(self):
        """Test segments marked as iscrowd. Allow only one per class per image"""

        for ds_set in self.sets:
            data = self.sets[ds_set]

            with self.subTest(set=ds_set):
                for ann in data['annotations']:

                    # Classes that have at least one iscrowd segment
                    iscrowd_cls = set()
                    for seg in ann['segments_info']:
                        if seg['iscrowd'] == 1:
                            self.assertTrue(seg['category_id'] not in iscrowd_cls, ann['file_name'])
                            iscrowd_cls.add(seg['category_id'])


    def test_stuff(self):
        """Test that all stuff classes have only one instance per image."""

        for ds_set in self.sets:
            data = self.sets[ds_set]

            stuff_categories = {cat['id'] for cat in data['categories'] if cat['isthing'] == 0}

            with self.subTest(set=ds_set):
                for ann in data['annotations']:

                    # Stuff classes that have already been found
                    done = set()
                    for seg in ann['segments_info']:
                        if seg['category_id'] in stuff_categories:
                            self.assertTrue(seg['category_id'] not in done, ann['file_name'])
                            done.add(seg['category_id'])


if __name__ == '__main__':
    unittest.main()

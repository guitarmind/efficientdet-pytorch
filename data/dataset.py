""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import random


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, root, ann_file, transform=None):
        super(CocoDetection, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.yxyx = True   # expected for TF model, most PT are xyxy
        self.include_masks = False
        self.include_bboxes_ignore = False
        self.has_annotations = 'image_info' not in ann_file
        self.coco = None
        self.cat_ids = []
        self.cat_to_label = dict()
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)

        self.mosaic_border = [-1024 // 2, -1024 // 2]

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            if valid_annotation and min(info['width'], info['height']) >= 32:
                self.img_ids.append(img_id)
                self.img_infos.append(info)
            else:
                self.img_ids_invalid.append(img_id)

    def _parse_img_ann(self, img_id, img_info):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            # To subtract 1 or not, TF doesn't appear to do this so will keep it out for now.
            if self.yxyx:
                #bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_to_label[ann['category_id']] if self.cat_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(img_id=img_id, bbox=bboxes, cls=cls, img_size=(img_info['width'], img_info['height']))

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann

    def load_mosaic(self, index, s):
        # loads images in a mosaic
        labels4 = []
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = load_image(self, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

            # Replicate
            # img4, labels4 = replicate(img4, labels4)

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = random_affine(img4, labels4,
                                      degrees=self.hyp['degrees'],
                                      translate=self.hyp['translate'],
                                      scale=self.hyp['scale'],
                                      shear=self.hyp['shear'],
                                      border=self.mosaic_border)  # border to remove

        return img4, labels4

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_id = self.img_ids[index]
        img_info = self.img_infos[index]
        if self.has_annotations:
            ann = self._parse_img_ann(img_id, img_info)
        else:
            ann = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            # markpeng - TODO
            # https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/f25ef5fdf4e5e62080fa88d29129dc55cbd2725e/dataset.py
            img, ann = self.transform(img, ann)

        return img, ann

    def __len__(self):
        return len(self.img_ids)

# encoding: utf-8
"""
@author: Xiaodong Wu
@version: 1.0
@file: csm.py
@time:  2019-06-05 20:30

"""
import glob
import re

import os
import os.path as osp


class CSM(object):
    """
    CMS dataset, used for WIDER FACE cast search challenge
    Reference:

    Dataset statistics:
    # identities: # in total, we ignore images with 'other' identity
    # images: * (train) + * (query) + * (gallery)
    """

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(CSM, self).__init__()
        self.images_dir = osp.join(root)
        self.train_path = 'train_bbox'
        self.gallery_path = 'val_bbox'
        self.query_path = 'val_bbox'

        self.root = root
        self.train_dir = osp.join(root, 'train_bbox')
        self.query_dir = osp.join(root, 'val_bbox')
        self.gallery_dir = osp.join(root, 'val_bbox')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, is_train=True)
        query = self._process_dir(self.query_dir, relabel=False, is_train=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, is_train=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_ids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_ids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_ids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, is_train=True):
        img_paths = []
        for movie_dir in os.listdir(dir_path):
            image_dir = os.path.join(dir_path, '{}/candidates'.format(movie_dir))
            img_paths_part = glob.glob(osp.join(image_dir, 'nm*.jpg'))
            img_paths.extend(img_paths_part)

        if not is_train:
            img_paths = img_paths[:min(len(img_paths), 10000)]  # use 10k images for evaluation,

        pid_container = set()
        dataset = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, movie_id, _, _ = img_name.split('_')
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid, movie_id, _, _ = img_name.split('_')
            if relabel:
                pid = pid2label[pid]
            relevant_img_path = '/'.join(img_path.split('/')[-3:])
            # dataset.append((img_path, pid, movie_id))
            dataset.append((relevant_img_path, pid, movie_id))
        return dataset

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

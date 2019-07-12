#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:05:54 2019

@author: wangruikui
"""

import time
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import six
import io
import json

class JsonDataset(data.Dataset):

    def __init__(self, BBOX_DIR, json_file, rich_transforms, state = 'can'):
        '''
        Args:
          root_folder: the root folder of images
          list_file: relative path of images, and their corresponding label(s)
        '''
        # read heatmap and optical path
        self.bbox_folder = BBOX_DIR
        self.json_file = json_file
        # read list
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        self.transforms = rich_transforms
        self.cans = []
        self.state = state
        if state == 'can':
            for idd, (key, movie) in enumerate(self.data.items()):
                for can in movie['candidates']:
                    self.cans.append(can)
        else :
            for idd, (key, movie) in enumerate(self.data.items()):
                for can in movie['cast']:
                    self.cans.append(can)

    def __getitem__(self, idx):
        '''
        load images and labels, and encode them to a 3-channel input and an output
        :param idx: (int) image index
        :return:
            input: an image
            output: the ground truth label
        '''
        # load images
        if self.state == 'can':
            try:
                img_name = '/'.join(self.cans[idx]['img'].split('/')[:-1]) + '/'+self.cans[idx]['label'] + '_' + self.cans[idx]['id'] + '_' + self.cans[idx]['img'].split('/')[-1]
            except:
                img_name = '/'.join(self.cans[idx]['img'].split('/')[:-1]) + '/'+ '_' + self.cans[idx]['id'] + '_' + self.cans[idx]['img'].split('/')[-1]

            Img = Image.open(os.path.join(self.bbox_folder,img_name))
        else:
            Img = Image.open(os.path.join(self.bbox_folder,self.cans[idx]['img']))
        if Img.mode is not 'RGB':
            Img = Img.convert('RGB')
        new_h = 384
        new_w = 256
        Img = Img.resize((new_h, new_w), Image.BILINEAR)
        if self.transforms:
            data = self.transforms(Img)
        else:
            Img = np.asarray(Img)
            data = torch.from_numpy(Img)

        # load groundtruth
        return data, self.cans[idx]['id']

    def __len__(self):
        return len(self.cans)


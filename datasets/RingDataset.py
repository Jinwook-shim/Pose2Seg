import torch
from torch.utils import data
from pathlib import Path
import glob
import json
import cv2
import numpy as np
class RingDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path_of_rgbs):
        self.path_of_rgbs = path_of_rgbs
        # self.path_of_rgbs = r'C:\Erez.Posner_to_NAS\HitMe_openpose_example\Components\RGB'
        list_IDs = glob.glob(str(Path(self.path_of_rgbs) / '*.png' ))
        # 'Initialization'
        self.openpose_to_coco_indices = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

        # self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        self.valid = 1
        self.ID = Path(self.list_IDs[index]).stem
        with open(str(Path(self.list_IDs[index]).parent.parent / r'coco_skeleton\{}_keypoints.json'.format(self.ID))) as json_file:
            data = json.load(json_file)
        self.img = cv2.imread(str(Path(self.list_IDs[index]).parent.parent / r"RGB\{}.png".format(self.ID)), -1)
        try:
            kp = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            self.kp = kp[self.openpose_to_coco_indices, :]
        except:
            self.kp=np.zeros(1)
            self.valid = 0
        self.mask = np.all(self.img > 0, 2).astype(np.uint8)[np.newaxis, ...]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]

        return self.img, self.kp,self.mask,self.ID,self.valid
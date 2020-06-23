import os
import sys
import multiprocessing as mp
from functools import partial

import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
from tqdm import tqdm

def crop_img(img, w=128, h=128, N=0, border=4, width=10, height=10):
    patches = np.zeros((N, height, width))
    
    for j in range(N):
        x = np.random.randint(border, w - width - border)
        y = np.random.randint(border, h - height - border)

        crop = img[x:x+width, y:y+height].copy()
        
        patches[j, :, :] = crop - crop.mean()
        
    return patches

class FolderPatchDataset(Dataset):

    def __init__(self, width:int, height:int, N:int=0, border:int=4, folder:str='./', fmt:str='.bmp', training=True):
        super(FolderPatchDataset, self).__init__()
        self.N = N
        self.width = width
        self.height = height
        self.border = border
        self.folder = folder
        self.fmt = fmt
        self.training = training
        
        # holder
        self.images = None
        
        # initialize patches
        self.extract_patches_()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    def extract_patches_(self):
        filenames = os.listdir(self.folder)
        image_filenames = []
        
        for filename in filenames:
            if self.fmt in filename:
                image_filenames.append(filename)
                
        n_img = len(image_filenames)
        
        imgs = [cv2.imread(os.path.join(self.folder, filename)) for filename in image_filenames]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        
        h, w = imgs[0].shape
        
        self.images = torch.zeros((self.N * n_img, self.width, self.height))
        
        counter = 0
        
        if self.training:
            crop_wrapper = partial(
                crop_img,
                w=w, h=h,
                N=self.N,
                border=self.border,
                width=self.width, height=self.height
            )
            
            pool = mp.Pool()

            self.images = list(tqdm(
                pool.imap_unordered(crop_wrapper, imgs),
                total=len(imgs),
                file=sys.stdout
            ))
            
            self.images = torch.from_numpy(np.concatenate(self.images))
        else:
            for img in tqdm(imgs):
                cols = (w - 2 * self.border) // self.width
                rows = (h - 2 * self.border) // self.height

                self.N = cols * rows

                for i in range(cols):
                    for j in range(rows):
                        x = self.border + i * self.width
                        y = self.border + j * self.height

                        crop = torch.from_numpy(img[x:x+self.width, y:y+self.height])

                        # whitten
                        self.images[counter, :, :] = crop - crop.mean()

                        counter += 1

import os
import sys
import multiprocessing as mp
from functools import partial

import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
from tqdm import tqdm

def random_crop_img(img, w=128, h=128, N=0, border=4, width=10, height=10, whiten=False):
    patches = np.zeros((N, height, width))
    
    for j in range(N):
        x = np.random.randint(border, w - width - border)
        y = np.random.randint(border, h - height - border)

        crop = img[y:y+height, x:x+width].copy()
        
        if whiten:
            patches[j, :, :] = crop - crop.mean()
        else:
            patches[j, :, :] = crop
        
    return patches

def crop_img(img, w=128, h=128, N=0, border=4, width=10, height=10, whiten=False):
    patches = np.zeros((N, height, width))
    
    cols = (w - 2 * border) // width
    rows = (h - 2 * border) // height
    
    for i in range(rows):
        for j in range(cols):
            crop = img[
                i*height+border:(i+1)*height+border,
                j*width+border:(j+1)*width+border
            ]
            
            if whiten:
                patches[i*cols+j, :, :] = crop - crop.mean()
            else:
                patches[i*cols+j, :, :] = crop
        
    return patches

class FolderPatchDataset(Dataset):

    def __init__(self, width:int, height:int, N:int=0, border:int=4, folder:str='./', fmt:str='.bmp', training=True, whiten=False):
        super(FolderPatchDataset, self).__init__()
        self.N = N
        self.width = width
        self.height = height
        self.border = border
        self.folder = folder
        self.fmt = fmt
        self.training = training
        self.whiten = whiten
        
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
        
        self.image_filenames = image_filenames
        n_img = len(image_filenames)
        
        imgs = [cv2.imread(os.path.join(self.folder, filename)) for filename in image_filenames]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        
        h, w = imgs[0].shape
        self.h = h
        self.w = w
        
        cols = (w - 2 * self.border) // self.width
        rows = (h - 2 * self.border) // self.height
        self.cols = cols
        self.rows = rows
        self.N = cols * rows
        
        if self.training:
            crop_wrapper = partial(
                random_crop_img,
                w=w, h=h,
                N=self.N,
                border=self.border,
                width=self.width, height=self.height,
                whiten=True
            )
            
            pool = mp.Pool()

            images = list(tqdm(
                pool.imap(crop_wrapper, imgs),
                total=len(imgs),
                file=sys.stdout
            ))
            
            self.images = torch.from_numpy(np.concatenate(images))
        else:
            crop_wrapper = partial(
                crop_img,
                w=w, h=h,
                N=self.N,
                border=self.border,
                width=self.width, height=self.height,
                whiten=True
            )
            
            pool = mp.Pool()

            images = list(tqdm(
                pool.imap(crop_wrapper, imgs),
                total=len(imgs),
                file=sys.stdout
            ))
            
            self.images = torch.from_numpy(np.concatenate(images))
            
        self.images /= 255.
        
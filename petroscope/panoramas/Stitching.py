import os, shutil
import cv2
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc
from PIL import Image
import typing

import torch
import torchvision

from .functions import *

import kornia as K
import kornia.feature as KF

class Stitcher():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        # self.size = (480, 600)
        self.size = (600, 400)
    
    # def preprocess(self, img_names: List[str]) -> List[torch.Tensor]: 

    def Stitch(self, directory, outputfile='output.jpg', logger=None, verbose=True):
        start_time = time()
        
        pic_names = os.listdir(directory)
        n = len(pic_names)

        images = []
        for name in pic_names: 
            path = os.path.join(directory, name)
            img = Image.open(path).convert('L')
            # orig_size = img.size[::-1]
            # img = img.resize(self.size[::-1], resample=Image.Resampling.LANCZOS)
            orig_size = img.size
            img = img.resize(self.size, resample=Image.Resampling.LANCZOS)

            img = torchvision.transforms.functional.pil_to_tensor(img) 
            img = img.unsqueeze(dim=0)
            images.append(img)    

        batch1 = []
        batch2 = []
        for i in range(n-1):
            for j in range(i+1, n):
                batch1.append(images[i])
                batch2.append(images[j])
        
        batch1 = torch.cat(batch1) / 255.0
        batch2 = torch.cat(batch2) / 255.0
        
        all_corr = []
        batch_size = 10
        total_infer = n * (n - 1) // 2
        batch_num = (total_infer - 1) // batch_size + 1

        if verbose:
            print(f'img processing done - {time() - start_time:.4}s')
        if logger:
            logger['images num'].append(len(pic_names))
            logger['preproc time'].append(time() - start_time)
            
        s_time = time()
        
        for i in range(batch_num):
            input_dict = {
                "image0": batch1[batch_size * i : batch_size * (i + 1)].to(self.device),
                "image1": batch2[batch_size * i : batch_size * (i + 1)].to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)
            tmp = {
                'batch_indexes' : correspondences['batch_indexes'].detach().cpu(),
                'keypoints0' : correspondences['keypoints0'].detach().cpu(),
                'keypoints1' : correspondences['keypoints1'].detach().cpu(),
                'confidence' : correspondences['confidence'].detach().cpu()
            }
            all_corr.append(tmp)
            del correspondences
            torch.cuda.empty_cache()
    
        if verbose:
            print(f'LoFTR done - {time() - s_time:.4}s')
        if logger:
            logger['LoFTR time'].append(time() - s_time)
        s_time = time()

        
        inliers = []
        diff_corr = []
        for batch_corr in all_corr:
            for i in range(batch_size):
                idx = batch_corr['batch_indexes'] == i
                kp0 = batch_corr['keypoints0'][idx]
                kp1 = batch_corr['keypoints1'][idx]
                conf = batch_corr['confidence'][idx]
                diff_corr.append(np.concatenate([kp0, kp1, conf[..., None]], axis=-1))
        
        good_corrs = []
        for corrs in diff_corr:
            corrs = corrs[corrs[:, 4] > 0.9]
            good_corrs.append(corrs)

        Hs = [[None] * n for _ in range(n) ]
        num_matches = np.zeros((n, n))
        for i in range(n-1):
            for j in range(i+1, n):
                corrs = good_corrs.pop(0)
                num = corrs.shape[0]
                if num < 10:
                    continue
        
                # corrs[:, 0] *= orig_size[1] / self.size[1]
                # corrs[:, 2] *= orig_size[1] / self.size[1]
                # corrs[:, 1] *= orig_size[0] / self.size[0]
                # corrs[:, 3] *= orig_size[0] / self.size[0]

                corrs[:, 0] *= orig_size[0] / self.size[0]
                corrs[:, 2] *= orig_size[0] / self.size[0]
                corrs[:, 1] *= orig_size[1] / self.size[1]
                corrs[:, 3] *= orig_size[1] / self.size[1]
                
                num_matches[i][j] = num
                num_matches[j][i] = num
                Hs[i][j], mask_ij = cv2.findHomography(corrs[:, 0:2], corrs[:, 2:4], cv2.USAC_MAGSAC)
                Hs[j][i], mask_ji = cv2.findHomography(corrs[:, 2:4], corrs[:, 0:2], cv2.USAC_MAGSAC)
                inliers_ij = corrs[mask_ij.squeeze().astype('bool')]
                inli = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:50, :-1]
                inli = inli[inli[:, -1] > 0.99]
                inliers += [[i, j, *inl] for inl in inli]
                
    
        if verbose:
            print(f'RANSAC done - {time() - s_time:.4}s')
        if logger:
            logger['homography time'].append(time() - s_time)
            
        transforms = [np.eye(3) for i in range(n)]

        queryIdx = [i for i in range(n)]
        Idx = [i for i in range(n)]
        targetIdx = []
        
        pivot = np.argmax(num_matches.sum(axis=1))
        targetIdx.append(pivot)
        queryIdx.remove(pivot)
        Idx.remove(pivot)
        
        while queryIdx:
            a = num_matches[queryIdx, :][:, targetIdx]
            curr, best_neighb = np.unravel_index(np.argmax(a, axis=None), a.shape)
            H =  transforms[targetIdx[best_neighb]] @ Hs[Idx[curr]][targetIdx[best_neighb]]
            H /= H[2, 2]
            transforms[Idx[curr]] = H
            targetIdx.append(Idx[curr])
            queryIdx.remove(Idx[curr])
            Idx.pop(curr)

        s_time = time()
        if logger:
            logger['num inliers'].append(len(inliers))
    
        final_transforms, init_error, optim_error = optimization(transforms, inliers, pivot)

        if verbose:
            print(f'optimization done - {time() - s_time:.4}s')
        if logger:
            logger['optimization time'].append(time() - s_time)
            logger['initial error'].append(init_error)
            logger['optimized error'].append(optim_error)
        s_time = time()

        panorama_ans = transform_and_stitch(final_transforms, directory)
        final_pic = panorama_ans.astype('uint8') 
        cv2.imwrite(outputfile, final_pic)
        
        if verbose:
            print(f'stitching done - {time() - s_time:.4}s')
        if logger:
            logger['stitching time'].append(time() - s_time)
            logger['total time'].append(time() - start_time)
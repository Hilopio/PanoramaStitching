import os, shutil
import cv2
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc
from PIL import Image 

import torch
import torchvision

from functions import *

import kornia as K
import kornia.feature as KF

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gc.collect()
torch.cuda.empty_cache()

matcher = KF.LoFTR(pretrained="outdoor").to(device)

new_size = (480, 600)

def Stich(directory, outputfile='output.jpg', logger=None, verbose=True):
    start_time = time()
    
    pic_names = os.listdir(directory)
    n = len(pic_names)

    images = []
    for name in pic_names: 
        path = os.path.join(directory, name)
        img = Image.open(path).convert('L')
        orig_size = img.size[::-1]
        
        img = img.resize(new_size[::-1], resample=Image.Resampling.LANCZOS)
        img = torchvision.transforms.functional.pil_to_tensor(img) 
        img = img.unsqueeze(dim=0)
        images.append(img)    

    gc.collect()
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
            "image0": batch1[batch_size * i : batch_size * (i + 1)].to(device),
            "image1": batch2[batch_size * i : batch_size * (i + 1)].to(device),
        }
        with torch.inference_mode():
            correspondences = matcher(input_dict)
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
    matches = [[None] * n for _ in range(n) ]
    num_matches = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            corrs = good_corrs.pop(0)
            num = corrs.shape[0]
            if num < 10:
                continue
    
            corrs[:, 0] *= orig_size[1] / new_size[1]
            corrs[:, 2] *= orig_size[1] / new_size[1]
            corrs[:, 1] *= orig_size[0] / new_size[0]
            corrs[:, 3] *= orig_size[0] / new_size[0]
            
            matches[i][j] = [corrs[:, 0:2], corrs[:, 2:4]] #?
            matches[j][i] = [corrs[:, 2:4], corrs[:, 0:2]] #?
            num_matches[i][j] = num
            num_matches[j][i] = num
            Hs[i][j], mask_ij = cv2.findHomography(corrs[:, 0:2], corrs[:, 2:4], cv2.USAC_MAGSAC)
            Hs[j][i], mask_ji = cv2.findHomography(corrs[:, 2:4], corrs[:, 0:2], cv2.USAC_MAGSAC)
            inliers_ij = corrs[mask_ij.squeeze().astype('bool')]
            inliers_ji = corrs[mask_ji.squeeze().astype('bool')]
            inli = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:50, :-1]
            inli = inli[inli[:, -1] > 0.99]
            # inli = inliers_ij[inliers_ij[:, -1] > 0.99, :-1]
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
    # final_pic = cv2.cvtColor(final_pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outputfile, final_pic)
    
    if verbose:
        print(f'stitching done - {time() - s_time:.4}s')
    if logger:
        logger['stitching time'].append(time() - s_time)
        logger['total time'].append(time() - start_time)


def execute_on_all(in_global_dir, out_dir):
    """_summary_

    Args:
        in_global_dir (str): path/name of global directory with datasets,
                            each dataset is a directory with images
        out_dir (str): path/name of directory with panoramas
    """
    if os.path.exists(out_dir):
        for dir in os.listdir(out_dir):
            file_path = os.path.join(out_dir, dir)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(out_dir)
    
    logger = {
        "series name"       : [],
        "images num"        : [],
        "preproc time"      : [],
        "LoFTR time"        : [],
        "homography time"   : [],
        "optimization time" : [],
        "stitching time"    : [],
        "total time"        : [],
        "num inliers"       : [],
        "initial error"     : [],
        "optimized error"   : []   
    }
    datasets = os.listdir(in_global_dir)
    for d in tqdm(datasets):
        in_dir = os.path.join(in_global_dir, d)
        out_file = os.path.join(out_dir, d + '.jpg')
        logger["series name"].append(d)
        Stich(in_dir, out_file, logger=logger, verbose=False)
    
    df = pd.DataFrame(logger)
    df.to_csv("info.csv")


if __name__ == '__main__':
    in_global_dir = os.path.join('LumenStone', 'data-calibrated')
    out_dir = os.path.join('LumenStone', 'panoramas-calibrated')
    execute_on_all(in_global_dir, out_dir)
import pickle
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from utils import _load_images, _warp, _warp_img
from graph_cutting import coarse_to_fine_optimal_seam

def find_overlap_region(mask1, mask2, eps=200):
    h, w = mask1.shape[:2]
    overlap_mask = mask1 & mask2
    overlap_idx = np.nonzero(overlap_mask)
    assert overlap_idx[0].size != 0, "нет области пересечения"

    y_min, y_max = np.min(overlap_idx[0]), np.max(overlap_idx[0])
    x_min, x_max = np.min(overlap_idx[1]), np.max(overlap_idx[1])
    Y_MIN, Y_MAX = max(y_min - eps, 0), min(y_max + eps, h),
    X_MIN, X_MAX = max(x_min - eps, 0), min(x_max + eps, w)

    small_window_slice = slice(y_min, y_max), slice(x_min, x_max)
    wide_window_slice = slice(Y_MIN, Y_MAX), slice(X_MIN, X_MAX)
    small_in_wide_slice = slice(y_min-Y_MIN, y_max-Y_MIN), slice(x_min-X_MIN, x_max-X_MIN)

    slices = (
        small_window_slice,
        wide_window_slice,
        small_in_wide_slice
    )
    return slices

def build_blured_stack(img, sigmas):
    return [img] + [cv2.GaussianBlur(img, (0, 0), sigma) for sigma in sigmas]

def build_bands_stack(blured_stack):
    bands_stack = []
    for i in range(len(blured_stack)-1):
        bands_stack.append(blured_stack[i] - blured_stack[i+1])
    bands_stack.append(blured_stack[-1])
    return bands_stack

def multi_band_blending(images, masks, levels):
    sigmas = [2.0 ** k  for k in range(levels - 1)]
    patches_bands = [build_bands_stack(build_blured_stack(img, sigmas)) for img in images]
    patches_weights = [build_blured_stack(mask.astype('float32'), sigmas) for mask in masks]

    patches_bands = list(zip(*patches_bands))
    patches_weights = list(zip(*patches_weights))
    
    pano_bands = []
    size = images[0].shape
    for img_bands, img_weights in zip(patches_bands, patches_weights):
        curr_band = np.zeros(size)
        curr_weights = np.zeros(size[:-1])
        for img, weight in zip(img_bands, img_weights):
            curr_band += img * weight[..., np.newaxis]
            curr_weights += weight
        pano_bands.append(curr_band / (curr_weights[..., np.newaxis] + 1e-6))
    return np.sum(pano_bands, axis=0)

def get_graphcut_mask(images, transforms, panorama_size):
    n = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)
    img_indexes = pano_mask.astype('int8') - np.ones_like(pano_mask, dtype='int8')

    for i in range(1, n):
        warped_pic, curr_mask = _warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, curr_mask)

        inter1 = pano[wide_window_slice]
        inter2 = warped_pic[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter1, inter2, small_in_wide_slice,
                                            coarse_scale=16, fine_scale=4, lane_width=200)

        curr_mask[wide_window_slice] = np.where(labels, False, True)
        
        pano = np.where(curr_mask[..., np.newaxis], warped_pic, pano)
        img_indexes = np.where(curr_mask, i, img_indexes)
        pano_mask = curr_mask | pano_mask

    return img_indexes

def _warp_pano_blended(images, transforms, panorama_size):
    n = len(images)
    img_indexes = get_graphcut_mask(images, transforms, panorama_size)
    masks = [np.where(img_indexes == i, True, False) for i in range(n) ]
    warpeds = [_warp_img(image, H, panorama_size) for image, H in zip(images, transforms)]
    pano = multi_band_blending(warpeds, masks, levels=7)
    return pano


def stitch_pano(transforms_file, output_file):

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)
        transforms = loaded_data["transforms"]
        panorama_size = loaded_data["panorama_size"]
        img_paths = loaded_data["img_paths"]

    pics = _load_images(img_paths)

    pano = _warp_pano_blended(pics, transforms, panorama_size)
    
    pano = (pano.clip(0, 1) * 255).astype(np.uint8)
    output_img = Image.fromarray(pano)
    output_img.save(output_file, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transforms_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    transforms_dir = Path(args.transforms_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if transforms_dir.exists() and transforms_dir.is_dir():
        for transforms_file in tqdm(transforms_dir.iterdir()):

            pano_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))
            stitch_pano(transforms_file, pano_file)

    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")
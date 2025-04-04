import pickle
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from utils import _load_images, _warp, _warp_img
from stitching.stitch_graphcut import coarse_to_fine_optimal_seam
from stitching.stitch_collage_gaincomp import gain_compensation, find_mean_color, compensate_mean_color

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

def get_gaussian_level(img, sigmas, k):
    if k == 0:
        return img
    else:
        return cv2.GaussianBlur(img, (0, 0), sigmas[k-1], borderType=cv2.BORDER_REPLICATE)

def get_laplacian_level(img, sigmas, k):
    if k == len(sigmas):
        return get_gaussian_level(img, sigmas, k)
    else:
        return get_gaussian_level(img, sigmas, k) - get_gaussian_level(img, sigmas, k + 1)


def multi_band_blending(images, masks, transforms,panorama_size, levels):
    sigmas = [2.0 ** k  for k in range(levels - 1)]

    w, h = panorama_size
    pano = np.zeros((h, w, 3))
    n_images = len(images)
    for k in tqdm(range(levels), desc='Levels', position=1):
        curr_band = np.zeros((h, w, 3))
        curr_weights = np.zeros((h, w))
        for i in tqdm(range(n_images), desc='Images', leave=False, position=2):
            warped_image = _warp_img(images[i], transforms[i], panorama_size)
            weights = get_gaussian_level(masks[i].astype('float32'), sigmas, k)
            band = get_laplacian_level(warped_image, sigmas, k)
            curr_band += band * weights[..., np.newaxis]
            curr_weights += weights
            del warped_image, weights, band
        pano += curr_band / (curr_weights[..., np.newaxis] + 1e-6)
    return pano

def find_graphcut_mask(images, transforms, panorama_size): 
    n = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)
    img_indexes = pano_mask.astype('int8') - np.ones_like(pano_mask, dtype='int8')
    
    for i in tqdm(range(1, n), desc='GraphCut', position=1):
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

        del warped_pic, curr_mask, labels, inter1, inter2
    del pano, pano_mask
    
    return img_indexes

def _warp_pano_blended(images, transforms, panorama_size):
    n = len(images)
    img_indexes = find_graphcut_mask(images, transforms, panorama_size)
    masks = [img_indexes == i for i in range(n) ]
    pano = multi_band_blending(images, masks, transforms, panorama_size, 7)
    return pano


def stitch_pano(transforms_file, output_file):

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)
        transforms = loaded_data["transforms"]
        panorama_size = loaded_data["panorama_size"]
        img_paths = loaded_data["img_paths"]

    images = _load_images(img_paths)

    target_mean_color = find_mean_color(images)
    images = gain_compensation(images, transforms, panorama_size)
    new_mean_color = find_mean_color(images)
    color_scale = target_mean_color / (new_mean_color + 1e-6)
    images = [compensate_mean_color(img, color_scale) for img in images]

    pano = _warp_pano_blended(images, transforms, panorama_size)
    
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
        for transforms_file in tqdm(transforms_dir.iterdir(), desc="Panoramas", position=0):

            pano_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))
            stitch_pano(transforms_file, pano_file)

    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")
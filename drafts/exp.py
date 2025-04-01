import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle

borderValue = 1.0


def warp_img(img, H, panorama_size):
    warped_img = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        flags=cv2.INTER_LINEAR,  # улучшить интерполяцию
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(borderValue, borderValue, borderValue)
    )
    return warped_img


def warp_mask(mask, H, panorama_size):
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        panorama_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype('bool')
    return warped_mask


def warp(image, H, panorama_size):
    warped_mask = warp_mask(np.ones(image.shape[:-1], dtype=int), H, panorama_size)
    warped_img = warp_img(image, H, panorama_size)
    return warped_img, warped_mask

def save_current(img, history_dir, name):
    path = history_dir / Path(name + '.jpg')
    img = (img.clip(0, 1) * 255).astype('uint8')
    Image.fromarray(img).save(path)

def _load_images(img_paths):
    images = []
    for path in img_paths:
        img = Image.open(path)
        img = np.array(img).astype(np.float32) / 255
        images.append(img)

    return images

def stitch_collage(images, transforms, panorama_size):
    panorama = np.zeros((*panorama_size[::-1], 3), dtype=np.float32)
    for image, H in zip(images, transforms):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    return (panorama * 255).astype('uint8')

def calculate_avg_colors(targetImg, queryImg, overlap_mask, gamma):
    targetImg_linear = targetImg ** (1 / gamma)
    queryImg_linear = queryImg ** (1 / gamma)

    target_overlap = targetImg_linear[overlap_mask]
    query_overlap = queryImg_linear[overlap_mask]

    targetColor = np.mean(target_overlap, axis=0)
    queryColor = np.mean(query_overlap, axis=0)

    return targetColor, queryColor

def stitch_experiment(images, transforms, panorama_size, history_dir):
    gamma = 2.2
    panorama, panorama_mask = warp(images[0], transforms[0], panorama_size)
    save_current(panorama, history_dir, '0')

    for i in tqdm(range(1, len(images))):
        warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
        overlap_mask = panorama_mask & warped_mask

        panorama_copy = np.where(warped_mask[..., np.newaxis], warped_img, panorama)
        save_current(panorama_copy, history_dir, f'{i}a')

        panoColor, warpedColor = calculate_avg_colors(panorama, warped_img, overlap_mask, gamma)
        warped_img = (warped_img ** (1 / gamma) - warpedColor + panoColor).clip(0, 1) ** gamma
        warped_img = np.where(warped_mask[..., np.newaxis], warped_img, borderValue)

        panorama_mask = panorama_mask | warped_mask
        panorama = np.where(warped_mask[..., np.newaxis], warped_img, panorama)
        save_current(panorama, history_dir, f'{i}b')

    # color_scales = np.array(color_scales)
    # global_comp_scale = color_scales.sum(axis=0) / (color_scales ** 2).sum(axis=0)

    # panorama = np.where(
    #     panorama_mask[..., np.newaxis],
    #     panorama * global_comp_scale ** (1 / gamma),
    #     (borderValue, borderValue, borderValue)
    # )


if __name__ == '__main__':
    transforms_file = Path('/home/g.nikolaev/pano/data/P1-transforms/008-data.pkl')
    history_dir = Path('/home/g.nikolaev/pano/data/008-history-mul')

    if history_dir.exists():
        shutil.rmtree(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)
        transforms = loaded_data["transforms"]
        panorama_size = loaded_data["panorama_size"]
        img_paths = loaded_data["img_paths"]

    pics = _load_images(img_paths)

    stitch_experiment(pics, transforms, panorama_size, history_dir)
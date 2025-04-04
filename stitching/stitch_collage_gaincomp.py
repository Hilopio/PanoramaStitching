import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
import shutil
import argparse
from scipy.optimize import least_squares, Bounds

from utils import _load_images, _warp_collage

borderValue = 0.0

def warp_img(img, H, panorama_size):
    warped_img = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        flags=cv2.INTER_NEAREST,  # улучшить интерполяцию
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

def find_mean_color(images):
    share_of_borders = 0.3
    maen_colors = []
    for img in images:
        height, width = img.shape[:-1]
        center_slise = (
            slice(int(height * share_of_borders), height - int(height * share_of_borders)),
            slice(int(width * share_of_borders), width - int(width * share_of_borders))
        )
        mean_color = np.mean(img[center_slise], axis=(0, 1))
        maen_colors.append(mean_color)
    return np.array(maen_colors).mean(axis=0)

def compensate_mean_color(image, color_scale):
    assert color_scale.shape == (3,)
    image = image * color_scale
    return image

def save_current(img, history_dir, name):
    path = history_dir / Path(name + '.jpg')
    img = (img.clip(0, 1) * 255).astype('uint8')
    Image.fromarray(img).save(path)

def fun(g, Sums):
    n = len(Sums)
    output = []
    for i in range(n):
        for j in range(i+1, n):
            output.append((1 + g[i]) * Sums[i][j] - (1 + g[j]) * Sums[j][i])
    return np.array(output)

def gain_compensation(images, transforms, panorama_size):
    n = len(images)
    warpeds, masks = [], []
    Sums = [[None] * n for _ in range(n)]

    for i in range(n):
        warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)
    
    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            Sums[i][j] = warpeds[i][mask].sum()
            Sums[j][i] = warpeds[j][mask].sum()
    
    g = np.zeros(n, dtype=np.float32)
    bounds = Bounds([-0.3] * n, [0.3] * n)
    res = least_squares(fun, g, method="trf", xtol=1e-10, ftol=1e-10, args=(Sums,), bounds=bounds)
    images = [img * (1 + res.x[i]) for i, img in enumerate(images)]
    return images

def stitch_collage_gaincomp(transforms_file, output_file):

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

    pano = _warp_collage(images, transforms, panorama_size)
    Image.fromarray(pano).save(output_file, quality=95)


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

            collage_gaincomp_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-gaincomp-collage.jpg"))
            stitch_collage_gaincomp(transforms_file, collage_gaincomp_file)

    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")
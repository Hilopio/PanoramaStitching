import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
from scipy.optimize import least_squares, Bounds

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


def fun(g, Sums):
    n = len(Sums)
    G = g.reshape((n, 3))
    output = []
    for i in range(n):
        for j in range(i+1, n):
            output.append((1 + G[i]) * Sums[i][j] - (1 + G[j]) * Sums[j][i])
    return np.array(output).reshape(-1)

def stitch_experiment(images, transforms, panorama_size, history_dir):
    gamma = 1
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
            Sums[i][j] = (warpeds[i][mask] ** gamma).sum(axis=0)
            Sums[j][i] = (warpeds[j][mask] ** gamma).sum(axis=0)
    
    g = np.zeros((n * 3), dtype=np.float32)
    bounds = Bounds([-0.3] * (3 * n), [0.3] * (3 * n))
    print('initial error = ',(fun(g, Sums) ** 2).mean() ** 0.5)
    res = least_squares(fun, g, method="trf", xtol=1e-6, ftol=1e-6, args=(Sums,), bounds=bounds)
    print('final error = ', (fun(res.x, Sums) ** 2).mean() ** 0.5)
    g = res.x.reshape((n, 3))

    images = [img * (1 + g[i]).astype('float32') for i, img in enumerate(images)]
    # corrected_images = []
    # for i in range(len(images)):
    #     img = images[i].copy()
    #     coeffs = g[i].copy()
    #     corrected_img = img * (1 + coeffs).astype('float32')  # Broadcasting: (H, W, 3) * (3,) → (H, W, 3)
    #     corrected_img = np.clip(corrected_img, 0, 1)
    #     corrected_images.append(corrected_img)

    pano = stitch_collage(images, transforms, panorama_size)
    Image.fromarray(pano).save(history_dir / '008-pano.jpg')


if __name__ == '__main__':
    transforms_file = Path('/home/g.nikolaev/pano/data/P1-transforms/008-data.pkl')
    history_dir = Path('/home/g.nikolaev/pano/data/collages2')

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
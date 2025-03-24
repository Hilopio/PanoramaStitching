import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
from scipy.optimize import least_squares, Bounds

from graph_cutting import coarse_to_fine_optimal_seam, find_overlap_region

borderValue = 0.0


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
    return (panorama.clip(0, 1) * 255).astype('uint8')

def warp_coarse_to_fine(images, transforms, panorama_size):
    n = len(images)
    panorama_ans = warp_img(images[0], transforms[0], panorama_size)

    for i in range(1, n):
        warped_pic = warp_img(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(panorama_ans, warped_pic)

        inter1 = panorama_ans[wide_window_slice]
        inter2 = warped_pic[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter1, inter2, small_in_wide_slice,
                                             coarse_scale=16, fine_scale=4, lane_width=200)

        mask1 = np.any(panorama_ans != borderValue, axis=2)
        panorama_ans = np.where(mask1[..., np.newaxis], panorama_ans, warped_pic)
        panorama_ans[wide_window_slice] = np.where(labels[..., np.newaxis], inter1, inter2)

    return (panorama_ans.clip(0, 1) * 255).astype('uint8')

def fun(g, Means, Numbers):
    n = len(Means)
    sigma_N = 10.0
    sigma_g = 0.1
    output = []
    for i in range(n):
        for j in range(i+1, n):
            output.append(np.sqrt(Numbers[i][j]) * (g[i] * Means[i][j] - g[j] * Means[j][i]) / sigma_N)
            output.append(np.sqrt(Numbers[i][j]) * (g[i] - 1.0) / sigma_g)
    return np.array(output)

def stitch_experiment(images, transforms, panorama_size, output_file):
    n = len(images)
    warpeds, masks = [], []
    Means = [[0] * n for _ in range(n)]
    Numbers = [[0] * n for _ in range(n)]

    for i in range(n):
        warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)
    
    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            if not mask.any():
                continue
            Means[i][j] = (warpeds[i][mask]).mean() * 255
            Numbers[i][j] = mask.sum()
            Means[j][i] = (warpeds[j][mask]).mean() * 255
            Numbers[j][i] = mask.sum()

    
    g = np.ones(n, dtype=np.float32)
    print('initial error = ',(fun(g, Means, Numbers) ** 2).mean() ** 0.5)
    res = least_squares(fun, g, method="lm", xtol=1e-6, ftol=1e-6, args=(Means, Numbers))
    print('final error = ', (fun(res.x, Means, Numbers) ** 2).mean() ** 0.5)
    g = res.x.reshape(n)
    images = [img * g[i] for i, img in enumerate(images)]

    # pano = stitch_collage(images, transforms, panorama_size)
    pano = warp_coarse_to_fine(images, transforms, panorama_size)
    Image.fromarray(pano).save(output_file, quality=95)

# def fun(g, Sums):
#     n = len(Sums)
#     G = g.reshape((n, 3))
#     output = []
#     for i in range(n):
#         for j in range(i+1, n):
#             output.append((1 + G[i]) * Sums[i][j] - (1 + G[j]) * Sums[j][i])
#     return np.array(output).reshape(-1)

# def stitch_experiment(images, transforms, panorama_size, output_file):
#     gamma = 1
#     n = len(images)
#     warpeds, masks = [], []
#     Sums = [[None] * n for _ in range(n)]

#     for i in range(n):
#         warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
#         warpeds.append(warped_img)
#         masks.append(warped_mask)
    
#     for i in range(n-1):
#         for j in range(i+1, n):
#             mask = masks[i] & masks[j]
#             Sums[i][j] = (warpeds[i][mask] ** gamma).sum(axis=0)
#             Sums[j][i] = (warpeds[j][mask] ** gamma).sum(axis=0)
    
#     g = np.zeros((n * 3), dtype=np.float32)
#     bounds = Bounds([-0.3] * (3 * n), [0.3] * (3 * n))
#     print('initial error = ',(fun(g, Sums) ** 2).mean() ** 0.5)
#     res = least_squares(fun, g, method="trf", xtol=1e-6, ftol=1e-6, args=(Sums,), bounds=bounds)
#     print('final error = ', (fun(res.x, Sums) ** 2).mean() ** 0.5)
#     g = res.x.reshape((n, 3))

#     images = [img * (1 + g[i]).astype('float32') for i, img in enumerate(images)]

#     # pano = stitch_collage(images, transforms, panorama_size)
#     pano = warp_coarse_to_fine(images, transforms, panorama_size)
#     Image.fromarray(pano).save(output_file, quality=95)

# def fun(g, Sums):
#     n = len(Sums)
#     output = []
#     for i in range(n):
#         for j in range(i+1, n):
#             output.append((1 + g[i]) * Sums[i][j] - (1 + g[j]) * Sums[j][i])
#     return np.array(output)

# def stitch_experiment(images, transforms, panorama_size, output_file):
#     gamma = 1
#     n = len(images)
#     warpeds, masks = [], []
#     Sums = [[None] * n for _ in range(n)]

#     for i in range(n):
#         warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
#         warpeds.append(warped_img)
#         masks.append(warped_mask)
    
#     for i in range(n-1):
#         for j in range(i+1, n):
#             mask = masks[i] & masks[j]
#             Sums[i][j] = (warpeds[i][mask] ** gamma).sum()
#             Sums[j][i] = (warpeds[j][mask] ** gamma).sum()
    
#     g = np.zeros(n, dtype=np.float32)
#     bounds = Bounds([-0.3] * n, [0.3] * n)
#     print('initial error = ',(fun(g, Sums) ** 2).mean() ** 0.5)
#     res = least_squares(fun, g, method="trf", xtol=1e-6, ftol=1e-6, args=(Sums,), bounds=bounds)
#     print('final error = ', (fun(res.x, Sums) ** 2).mean() ** 0.5)
#     g = res.x.copy()

#     images = [img * (1 + g[i]) for i, img in enumerate(images)]

#     # pano = stitch_collage(images, transforms, panorama_size)
#     pano = warp_coarse_to_fine(images, transforms, panorama_size)
#     Image.fromarray(pano).save(output_file, quality=95)


if __name__ == '__main__':
    transforms_dir = Path('../data/P1-transforms-ffc')
    output_dir = Path('../data/P1-gaincomp_graphcut-ffc')


    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for transforms_file in tqdm(transforms_dir.iterdir()):
        transforms_path = transforms_dir / transforms_file
        output_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))

        with open(transforms_file, "rb") as f:
            loaded_data = pickle.load(f)
            transforms = loaded_data["transforms"]
            panorama_size = loaded_data["panorama_size"]
            img_paths = loaded_data["img_paths"]

        pics = _load_images(img_paths)
        stitch_experiment(pics, transforms, panorama_size, output_file)
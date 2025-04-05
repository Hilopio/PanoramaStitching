import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle
from scipy.optimize import least_squares, Bounds

from utils import _load_images, _warp_collage, _warp

# def fun(g, Means, Numbers):
#     n = len(Means)
#     sigma_N = 10.0
#     sigma_g = 1
#     output = []
#     for i in range(n):
#         for j in range(i+1, n):
#             output.append(np.sqrt(Numbers[i][j]) * (g[i] * Means[i][j] - g[j] * Means[j][i]) / sigma_N)
#             output.append(np.sqrt(Numbers[i][j]) * (g[i] - 1.0) / sigma_g)
#     return np.array(output)

# def stitch_experiment(images, transforms, panorama_size, output_file):
#     n = len(images)
#     warpeds, masks = [], []
#     Means = [[0] * n for _ in range(n)]
#     Numbers = [[0] * n for _ in range(n)]

#     for i in range(n):
#         warped_img, warped_mask = warp(images[i], transforms[i], panorama_size)
#         warpeds.append(warped_img)
#         masks.append(warped_mask)

#     for i in range(n-1):
#         for j in range(i+1, n):
#             mask = masks[i] & masks[j]
#             if not mask.any():
#                 continue
#             Means[i][j] = (warpeds[i][mask]).mean() * 255
#             Numbers[i][j] = mask.sum()
#             Means[j][i] = (warpeds[j][mask]).mean() * 255
#             Numbers[j][i] = mask.sum()


#     g = np.ones(n, dtype=np.float32)
#     print('initial error = ',(fun(g, Means, Numbers) ** 2).mean() ** 0.5)
#     res = least_squares(fun, g, method="lm", xtol=1e-6, ftol=1e-6, args=(Means, Numbers))
#     print('final error = ', (fun(res.x, Means, Numbers) ** 2).mean() ** 0.5)
#     g = res.x.reshape(n)
#     print('g = ', g)
#     images = [img * g[i] for i, img in enumerate(images)]

#     pano = stitch_collage(images, transforms, panorama_size)
#     Image.fromarray(pano).save(output_file, quality=95)

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

#     pano = stitch_collage(images, transforms, panorama_size)
#     Image.fromarray(pano).save(output_file, quality=95)

def fun(g, Sums):
    n = len(Sums)
    output = []
    for i in range(n):
        for j in range(i+1, n):
            output.append((1 + g[i]) * Sums[i][j] - (1 + g[j]) * Sums[j][i])
    return np.array(output)


def stitch_experiment(images, transforms, panorama_size, output_file):
    gamma = 1
    n = len(images)
    warpeds, masks = [], []
    Sums = [[None] * n for _ in range(n)]

    for i in range(n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)
        warpeds.append(warped_img)
        masks.append(warped_mask)

    for i in range(n-1):
        for j in range(i+1, n):
            mask = masks[i] & masks[j]
            Sums[i][j] = (warpeds[i][mask] ** gamma).sum()
            Sums[j][i] = (warpeds[j][mask] ** gamma).sum()

    g = np.zeros(n, dtype=np.float32)
    bounds = Bounds([-0.3] * n, [0.3] * n)
    print('initial error = ', (fun(g, Sums) ** 2).mean() ** 0.5)
    res = least_squares(fun, g, method="trf", xtol=1e-6, ftol=1e-6, args=(Sums,), bounds=bounds)
    print('final error = ', (fun(res.x, Sums) ** 2).mean() ** 0.5)
    g = res.x.copy()

    images = [img * (1 + g[i]) for i, img in enumerate(images)]

    pano = _warp_collage(images, transforms, panorama_size)
    Image.fromarray(pano).save(output_file, quality=95)


if __name__ == '__main__':
    transforms_dir = Path('/home/g.nikolaev/pano/data/P1-bad-transforms')
    output_dir = Path('/home/g.nikolaev/pano/data/experiments-collages')

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for transforms_file in tqdm(transforms_dir.iterdir()):
        transforms_path = transforms_dir / transforms_file
        output_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-optim.jpg"))

        with open(transforms_file, "rb") as f:
            loaded_data = pickle.load(f)
            transforms = loaded_data["transforms"]
            panorama_size = loaded_data["panorama_size"]
            img_paths = loaded_data["img_paths"]

        pics = _load_images(img_paths)
        stitch_experiment(pics, transforms, panorama_size, output_file)

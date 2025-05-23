import gc
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import kornia.feature as KF
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import least_squares
# from tqdm import tqdm


def find_translation_and_panorama_size(sizes, transformations):
    x_coords = []
    y_coords = []
    for size, H in zip(sizes, transformations):
        corners = np.array([
            [0, 0, 1],
            [0, size[1], 1],
            [size[0], 0, 1],
            [size[0], size[1], 1]
        ])
        new_corners = H @ corners.T

        new_corners /= new_corners[2]
        x_coords += new_corners[0].tolist()
        y_coords += new_corners[1].tolist()

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]])

    panorama_size = (int(np.ceil(x_max - x_min)), int(np.ceil(y_max - y_min)))
    return T, panorama_size


def vec_to_homography(vec: np.ndarray, i: int, pivot: int) -> np.ndarray:
    # If the index is the pivot, return the identity matrix
    if i == pivot:
        return np.eye(3)
    # Adjust index if it is greater than pivot
    elif i > pivot:
        i -= 1
    # Extract the 3x3 homography matrix from the vector
    H = vec[8 * i: 8 * (i + 1)]
    H = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], 1]])
    return H


def homography_to_vec(Hs: List[np.ndarray], pivot: int) -> List[float]:
    n = len(Hs)
    vec = np.empty(8 * (n - 1))
    for i in range(n):
        if i == pivot:
            # Skip the pivot image
            continue
        elif i < pivot:
            # The homography matrix is placed at the position of the image
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * i: 8 * (i + 1)] = H
        else:
            # The homography matrix is placed at the position of the image
            # minus one (since the pivot image is skipped)
            H = Hs[i].reshape(-1)
            H = H[:-1]  # Remove the last element (scale factor)
            vec[8 * (i - 1): 8 * i] = H
    return vec


def dist(X: List[float], inliers: List[np.ndarray], pivot: int) -> np.ndarray:
    output = []  # Initialize the output list to store distances
    for i, j, x, y, xx, yy in inliers:
        # Get the homography matrices for images i and j
        Hi = vec_to_homography(X, i, pivot)
        Hj = vec_to_homography(X, j, pivot)

        # Transform the coordinates using the homography matrices
        first = np.dot(Hi, [x, y, 1])
        first /= first[2]  # Normalize to get the final coordinates
        second = np.dot(Hj, [xx, yy, 1])
        second /= second[2]  # Normalize to get the final coordinates
        output.append(first[0] - second[0])
        output.append(first[1] - second[1])

    return np.array(output)


def optimize(
    Hs: List[np.ndarray],
    inliers: List[np.ndarray],
    pivot: int,
) -> Tuple[List[np.ndarray], float, float]:

    n = len(Hs)
    vec = homography_to_vec(Hs, pivot)
    norm = dist(vec, inliers, pivot)

    init_error = (norm**2).mean() ** 0.5
    res_lm = least_squares(
        dist, vec, method="lm", xtol=1e-6, ftol=1e-6, args=(inliers, pivot)
    )
    optim_error = (res_lm.fun**2).mean() ** 0.5
    new_vec = res_lm.x

    final_transforms = []
    for i in range(n):
        final_transforms.append(vec_to_homography(new_vec, i, pivot))
    return final_transforms, init_error, optim_error


class Stitcher:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        self.matcher = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.size = np.array((600, 400))

    def _load_torch_tensors(
        self, img_paths: Iterable[Path]
    ) -> Tuple[List[Tuple[float]], List[torch.Tensor]]:
        images = []
        orig_sizes = []
        for path in img_paths:
            img = Image.open(path).convert("L")  # Convert to grayscale
            orig_sizes.append(np.array(img.size))  # Save the original size
            img = img.resize((600, 400), resample=Image.Resampling.LANCZOS)
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.unsqueeze(dim=0)
            images.append(img)  # Append the processed tensor to the list

        return orig_sizes, images

    def only_transforms(
        self,
        img_paths: Iterable[Path],
    ) -> Image:
        n = len(img_paths)
        orig_sizes, images = self._load_torch_tensors(img_paths)

        batch1 = []
        batch2 = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                batch1.append(images[i])
                batch2.append(images[j])

        batch1 = torch.cat(batch1) / 255.0
        batch2 = torch.cat(batch2) / 255.0

        all_corr = []
        batch_size = 10
        total_infer = n * (n - 1) // 2
        batch_num = (total_infer - 1) // batch_size + 1

        # Run the LoFTR model on the images
        for i in range(batch_num):
            input_dict = {
                "image0": batch1[batch_size * i: batch_size * (i + 1)].to(
                    self.device
                ),
                "image1": batch2[batch_size * i: batch_size * (i + 1)].to(
                    self.device
                ),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)
            tmp = {
                "batch_indexes": correspondences["batch_indexes"]
                .detach()
                .cpu(),
                "keypoints0": correspondences["keypoints0"].detach().cpu(),
                "keypoints1": correspondences["keypoints1"].detach().cpu(),
                "confidence": correspondences["confidence"].detach().cpu(),
            }
            all_corr.append(tmp)
            del correspondences
            torch.cuda.empty_cache()
            gc.collect()

        inliers = []
        diff_corr = []
        for batch_corr in all_corr:
            for i in range(batch_size):
                idx = batch_corr["batch_indexes"] == i
                if not idx.any():  # мб баг
                    break
                kp0 = batch_corr["keypoints0"][idx]
                kp1 = batch_corr["keypoints1"][idx]
                conf = batch_corr["confidence"][idx]
                diff_corr.append(
                    np.concatenate([kp0, kp1, conf[..., None]], axis=-1)
                )

        # Filter out the correspondences with low confidence
        good_corrs = []
        for corrs in diff_corr:
            corrs = corrs[corrs[:, 4] > 0.95]
            good_corrs.append(corrs)  # сделать фильтрацией

        Hs = [[None] * n for _ in range(n)]
        num_matches = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            for j in range(i + 1, n):
                corrs = good_corrs.pop(0)
                num = corrs.shape[0]
                if num < 10:
                    continue

                corrs[:, 0:2] *= orig_sizes[i] / self.size
                corrs[:, 2:4] *= orig_sizes[j] / self.size

                num_matches[i][j] = num
                num_matches[j][i] = num
                Hs[i][j], mask_ij = cv2.findHomography(
                    corrs[:, 0:2], corrs[:, 2:4], cv2.USAC_MAGSAC, 0.75
                )
                if mask_ij.shape[0] < 10:
                    continue

                # try:
                #     Hs[j][i] = np.linalg.inv(Hs[i][j])
                #     Hs[j][i] /= Hs[j][i][2, 2]
                # except:
                #     Hs = np.eye(3)

                Hs[j][i], mask_ji = cv2.findHomography(
                    corrs[:, 2:4], corrs[:, 0:2], cv2.USAC_MAGSAC, 0.75
                )

                inliers_ij = corrs[mask_ij.squeeze().astype("bool")]
                inli = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:15]
                inli = inli[:, :-1]
                inliers += [[i, j, *inl] for inl in inli]

                # inliers_ji = corrs[mask_ji.squeeze().astype("bool")]
                # inli = inliers_ji[inliers_ji[:, -1].argsort()[::-1]][:15]
                # inli = inli[:, :-1]
                # inliers += [[j, i, *inl[2:4], *inl[0:2]] for inl in inli]

        # Initialize the transformations for each image
        transforms = [np.eye(3) for i in range(n)]

        queryIdx = [i for i in range(n)]
        targetIdx = []

        pivot = np.argmax(num_matches.sum(axis=1))
        targetIdx.append(pivot)
        queryIdx.remove(pivot)

        while queryIdx:
            a = num_matches[queryIdx, :][:, targetIdx]
            # curr, best_neighb = np.unravel_index(
            #     np.argmax(a, axis=None), a.shape
            # )
            curr = np.argmax(a.sum(axis=1))
            best_neighb = np.argmax(a[curr])

            H = (
                transforms[targetIdx[best_neighb]]
                @ Hs[queryIdx[curr]][targetIdx[best_neighb]]
            )
            H /= H[2, 2]
            transforms[queryIdx[curr]] = H
            targetIdx.append(queryIdx[curr])
            queryIdx.pop(curr)

        # Optimize the transformations
        final_transforms, init_error, optim_error = optimize(
            transforms, inliers, pivot
        )

        T, panorama_size = find_translation_and_panorama_size(orig_sizes, final_transforms)
        final_transforms = [T @ H for H in final_transforms]

        # reordering
        final_transforms = [final_transforms[i] for i in targetIdx]
        new_img_paths = [img_paths[i] for i in targetIdx]
        return final_transforms, panorama_size, new_img_paths

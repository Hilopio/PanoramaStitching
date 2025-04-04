import gc
from pathlib import Path
from time import time
from typing import Iterable, List, Tuple, Union

import cv2
import argparse
import shutil
import kornia.feature as KF
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import least_squares
import maxflow

from tqdm import tqdm

borderValue = 255


def _load_torch_tensors(img_paths: Iterable[Path]) -> Tuple[List[Tuple[float]], List[torch.Tensor]]:
    images = []
    orig_sizes = []
    for path in img_paths:
        img = Image.open(path).convert("L")  # Convert to grayscale
        orig_sizes.append(np.array(img.size))  # Save the original size
        img = img.resize((600, 400), resample=Image.Resampling.LANCZOS)
        img = torchvision.transforms.functional.pil_to_tensor(img)
        img = img.unsqueeze(dim=0)
        images.append(img / 255)  # Append the processed tensor to the list

    return orig_sizes, images


def _load_images(img_paths: Iterable[Path]) -> Tuple[List[Tuple[float]], List[torch.Tensor]]:
    images = []
    for path in img_paths:
        img = Image.open(path)
        img = np.array(img).astype(np.float32) / 255
        images.append(img)

    return images


def compute_overlap_mask(img1, img2):
    mask1 = np.any(img1 != borderValue, axis=2)
    mask2 = np.any(img2 != borderValue, axis=2)
    return mask1 & mask2


def diff(img1, img2):
    diff = np.sum((img1 - img2) ** 2, axis=2)
    diff = np.sqrt(diff)
    return diff


def grad(img):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad


def GraphCut(img1, img2, only1_mask, only2_mask):
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(img1.shape[:-1])

    sigma = 1.0
    blurred_1 = cv2.GaussianBlur(img1, (0, 0), sigma)
    blurred_2 = cv2.GaussianBlur(img2, (0, 0), sigma)

    grad_1 = grad(blurred_1)
    grad_2 = grad(blurred_2)
    difference = diff(blurred_1, blurred_2)
    grad_difference = diff(grad_1, grad_2)

    alpha = 2
    smooth_map = difference + alpha * grad_difference
    g.add_grid_edges(nodeids, smooth_map, symmetric=True)

    left_inf = nodeids[only1_mask.astype(bool)]
    g.add_grid_tedges(np.array(left_inf), np.inf, 0)

    right_inf = nodeids[only2_mask.astype(bool)]
    g.add_grid_tedges(np.array(right_inf), 0, np.inf)

    g.maxflow()
    lbls_mask = g.get_grid_segments(nodeids)
    lbls_mask = np.int_(np.logical_not(lbls_mask))
    return lbls_mask


def find_overlap_region(img1, img2, eps=200):
    h, w = img1.shape[:2]
    overlap_mask = compute_overlap_mask(img1, img2)
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


def scaled_graph_cut(img1, img2, sure1, sure2, scale=2):
    if not sure2.any():
        return np.ones(img1.shape[:2], dtype='float32')
    orig_size = np.array((img1.shape[1], img1.shape[0]))
    new_size = (orig_size / scale).astype(int)
    smaller_1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_sure1 = cv2.resize(sure1.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)  # astype убрать?
    smaller_sure2 = cv2.resize(sure2.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)  # astype

    lbls_mask = GraphCut(smaller_1, smaller_2, smaller_sure1, smaller_sure2)

    lbls_mask = cv2.resize(lbls_mask, orig_size, interpolation=cv2.INTER_NEAREST)
    return lbls_mask.astype('float32')


def labels2seam(lbls_mask, width=7):
    dilated = cv2.dilate(lbls_mask, kernel=np.ones((width, width), dtype=np.uint8))
    seam = dilated - lbls_mask
    return seam


def seam2lane(seam_mask, width=200):
    dilated = cv2.dilate(seam_mask, kernel=np.ones((width, width), dtype=np.uint8))
    return dilated


def warp(img, H, panorama_size):
    warped = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(borderValue, borderValue, borderValue),
    )
    return warped


def coarse_to_fine_optimal_seam(img1, img2, small_in_wide_slice, coarse_scale=8, fine_scale=2, lane_width=200):
    overlap_mask = compute_overlap_mask(img1, img2)
    only1 = np.any(img1 != borderValue, axis=2) & np.logical_not(overlap_mask)
    only2 = np.any(img2 != borderValue, axis=2) & np.logical_not(overlap_mask)

    if only2.sum() < 0.01 * only2.size:
        return np.ones(img1.shape[:2], dtype='float32')

    coarse_labels = scaled_graph_cut(img1, img2, only1, only2, scale=coarse_scale)

    coarse_labels = coarse_labels[small_in_wide_slice]
    coarse_seam = labels2seam(coarse_labels, width=3)
    lane = seam2lane(coarse_seam, width=lane_width)

    sure1 = only1
    sure1[small_in_wide_slice] = (coarse_labels - lane).clip(0, 1)
    sure2 = only2
    sure2[small_in_wide_slice] = (1 - coarse_labels - lane).clip(0, 1)

    fine_labels = scaled_graph_cut(img1, img2, sure1, sure2, scale=fine_scale)

    return fine_labels


def find_warp_params(
    transforms: List[np.ndarray], img_paths: Iterable[Path]
) -> np.ndarray:
    pics = [cv2.imread(img_p).astype(np.float32) for img_p in img_paths]
    n = len(pics)

    # Initialize arrays to store original and transformed corner points
    all_corners = np.empty((n, 4, 3))
    for i in range(n):
        # Define corner points for each image
        all_corners[i] = [
            [0, 0, 1],
            [pics[i].shape[1], 0, 1],
            [pics[i].shape[1], pics[i].shape[0], 1],
            [0, pics[i].shape[0], 1],
        ]

    all_new_corners = np.empty((n, 4, 3))
    for i in range(n):
        # Apply homography transformations to each corner point
        all_new_corners[i] = [
            np.dot(transforms[i], corner) for corner in all_corners[i]
        ]

    # Reshape transformed corners for further processing
    all_new_corners = all_new_corners.reshape(-3, 3)
    x_news = all_new_corners[:, 0] / all_new_corners[:, 2]
    y_news = all_new_corners[:, 1] / all_new_corners[:, 2]

    # Determine min/max x and y coordinates for the panorama
    y_min = min(y_news)
    x_min = min(x_news)
    y_max = int(round(max(y_news)))
    x_max = int(round(max(x_news)))

    # Calculate shifts to adjust the panorama's origin
    x_shift = -min(x_min, 0)
    y_shift = -min(y_min, 0)
    T = np.array(
        [[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]], dtype="float32"
    )

    # Calculate new dimensions for the panorama
    x_min = int(round(x_min))
    y_min = int(round(y_min))
    height_new = y_max - y_min
    width_new = x_max - x_min
    size = (width_new, height_new)

    return T, size, pics


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
    """Extract a 3x3 homography matrix from a flattened vector.

    Args:
        vec (np.ndarray): Flattened vector of all homographies
            (except the pivot one, which is identical).

        i (int): The index of the homography to be extracted.

        pivot (int): The index of the pivot image.

    Returns:
        np.ndarray: The 3x3 homography matrix of the i-th image.
    """
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
    """
    Flatten a list of 3x3 homography matrices into a single vector.

    Args:
        Hs (List[np.ndarray]): A list of all homography matrices.
        pivot (int): The index of the pivot image.

    Returns:
        List[float]: A flattened vector of all homography matrices
        (except the pivot).
    """
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
    """
    Calculate distances between the coordinates of all pairs of inliers
    in the transformed coordinate system.

    Args:
        X (List[float]): Flattened vector of all homographies
        (except the pivot one, which is identity).

        inliers (List[np.ndarray]): List of inliers, each inlier is
        an np.ndarray((i, j, x, y, xx, yy)), where i and j are the indices
        of the images corresponding to the inlier, (x, y) are the coordinates
        of the point on image i, and (xx, yy) are the coordinates of the
        point on image j.

        pivot (int): Index of the pivot image.

    Returns:
        np.ndarray: A vector of distances between the coordinates of all
        pairs of inliers in the transformed coordinates.
    """
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
    """Global alignment using all inliers by adjusting all homography matrices.

    Args:
        Hs: a list of all homographies,

        inliers: list of inliers, each inlier is
        a np.ndarray((i, j, x, y, xx, yy)), where i and j are
        the indices of the images corresponding to the inlier,
        (x, y) are the coordinates of the point on image i,
        and (xx, yy) are the coordinates of the point on image j

        pivot: the number of the pivot image,

    Returns:
        a tuple of:
            * a list of new homographies,
            * the initial mean squared error,
            * the optimized mean squared error,
    """
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
        """
        Opens images from paths, saves the original dimensions,
        converts the images to the required format, and collects
        them into a list.

        Args:
            img_paths : paths to images

        Returns:
            A tuple of
                - list of original sizes of images
                - list of torch.Tensor of size [1, 1, 600, 400]
        """
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
        verbose: bool = False,
        logger: Union[bool, dict[str, list]] = False,
    ) -> Image:
        """
        A method that implements stitching a panorama from a collection
        of image files.

        Args:
            verbose : If True, it outputs information about the execution time of the stages
              and the accuracy of the stitching
            logger : A technical argument used for logging information about execution time
            and accuracy into a dictionary. Leave it as False
        """
        self.img_paths = img_paths
        start_time = time()
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

        if verbose:
            print(f"img processing done - {time() - start_time:.4}s")
        if logger:
            logger["images num"].append(n)
            logger["preproc time"].append(time() - start_time)
        s_time = time()

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

        if verbose:
            print(f"LoFTR done - {time() - s_time:.4}s")
        if logger:
            logger["LoFTR time"].append(time() - s_time)
        s_time = time()
        # Filter out the correspondences with low confidence
        inliers = []
        diff_corr = []
        for batch_corr in all_corr:
            for i in range(batch_size):
                idx = batch_corr["batch_indexes"] == i
                kp0 = batch_corr["keypoints0"][idx]
                kp1 = batch_corr["keypoints1"][idx]
                conf = batch_corr["confidence"][idx]
                kp0 *= orig_sizes[i] / self.size
                kp1 *= orig_sizes[i] / self.size
                diff_corr.append(
                    np.concatenate([kp0, kp1, conf[..., None]], axis=-1)
                )

        good_corrs = []
        for corrs in diff_corr:
            corrs = corrs[corrs[:, 4] > 0.9]
            good_corrs.append(corrs)

        Hs = [[None] * n for _ in range(n)]
        num_matches = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                corrs = good_corrs.pop(0)
                num = corrs.shape[0]
                if num < 10:
                    continue

                num_matches[i][j] = num
                num_matches[j][i] = num
                Hs[i][j], mask_ij = cv2.findHomography(
                    corrs[:, 0:2], corrs[:, 2:4], cv2.USAC_MAGSAC, 0.5
                )
                Hs[j][i], mask_ji = cv2.findHomography(
                    corrs[:, 2:4], corrs[:, 0:2], cv2.USAC_MAGSAC, 0.5
                )
                inliers_ij = corrs[mask_ij.squeeze().astype("bool")]

                inli = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:15]
                inli = inli[:, :-1]

                inliers += [[i, j, *inl] for inl in inli]

        if verbose:
            print(f"RANSAC done - {time() - s_time:.4}s")
        if logger:
            logger["homography time"].append(time() - s_time)

        # Initialize the transformations for each image
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
            # curr, best_neighb = np.unravel_index(
            #     np.argmax(a, axis=None), a.shape
            # )
            curr = np.argmax(a.sum(axis=1))
            best_neighb = np.argmax(a[curr])

            H = (
                transforms[targetIdx[best_neighb]]
                @ Hs[Idx[curr]][targetIdx[best_neighb]]
            )
            H /= H[2, 2]
            transforms[Idx[curr]] = H
            targetIdx.append(Idx[curr])
            queryIdx.remove(Idx[curr])
            Idx.pop(curr)

        s_time = time()
        if logger:
            logger["num inliers"].append(len(inliers))

        self.transforms = [transforms[i] for i in targetIdx]
        self.img_paths = [self.img_paths[i] for i in targetIdx]

        reverse_permute = [0] * len(targetIdx)
        for i, x in enumerate(targetIdx):
            reverse_permute[x] = i

        for i in range(len(inliers)):
            inliers[i][0] = reverse_permute[inliers[i][0]]
            inliers[i][1] = reverse_permute[inliers[i][1]]
        pivot = 0

        # Optimize the transformations
        final_transforms, init_error, optim_error = optimize(
            self.transforms, inliers, pivot
        )

        if verbose:
            print(f"optimization done - {time() - s_time:.4}s")
            print(f"num inliers - {len(inliers)}")
            print(f"initial error - {init_error:.4}")
            print(f"optimized error - {optim_error:.4}")
        if logger:
            logger["optimization time"].append(time() - s_time)
            logger["initial error"].append(init_error)
            logger["optimized error"].append(optim_error)
        return final_transforms


def mean_color(img, mask):
    masked_pixels = img[mask]
    return masked_pixels.mean(axis=0)


def shift_image_color(queryImg, queryMeanColor, targetMeanColor):
    bgMask = (queryImg == borderValue)
    newImg = queryImg + targetMeanColor[np.newaxis, np.newaxis, :] - queryMeanColor[np.newaxis, np.newaxis, :]
    newImg += bgMask.astype('float32') * borderValue
    return newImg.clip(0, 255)


def warp_coarse_to_fine(T, panorama_size, pics, transforms):
    n = len(pics)
    panorama_ans = warp(pics[0], T @ transforms[0], panorama_size)

    for i in tqdm(range(1, n)):
        warped_pic = warp(pics[i], T @ transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(panorama_ans, warped_pic)
        overlap_mask = compute_overlap_mask(warped_pic, panorama_ans)
        queryColor = mean_color(warped_pic, overlap_mask)
        targetColor = mean_color(panorama_ans, overlap_mask)
        warped_pic = shift_image_color(warped_pic, queryColor, targetColor)
        inter1 = panorama_ans[wide_window_slice]
        inter2 = warped_pic[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter1, inter2, small_in_wide_slice,
                                             coarse_scale=16, fine_scale=4, lane_width=200)

        mask1 = np.any(panorama_ans != borderValue, axis=2)
        panorama_ans = np.where(mask1[..., np.newaxis], panorama_ans, warped_pic)
        panorama_ans[wide_window_slice] = np.where(labels[..., np.newaxis], inter1, inter2)

    return panorama_ans


def stitch_pano(input_dir, output_file, stchr):
    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    transforms = stchr.only_transforms(img_paths=img_paths, verbose=False)
    T, size, pics = find_warp_params(stchr.transforms, stchr.img_paths)
    panorama_ans = warp_coarse_to_fine(T, size, pics, stchr.transforms)
    panorama_rgb = cv2.cvtColor(panorama_ans, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(panorama_rgb.astype("uint8"))
    output_img.save(output_file, quality=95)


if __name__ == '__main__':
    device = 'cuda:5'
    stchr = Stitcher(device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_global_dir', type=str)
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    input_global_dir = Path(args.input_global_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_global_dir.exists() and input_global_dir.is_dir():
        # Iterate over inner directories
        for inner_dir in tqdm(input_global_dir.iterdir()):
            if inner_dir.is_dir():
                output_file = output_dir / Path(inner_dir.name + '-pano0.jpg')
                stitch_pano(inner_dir, output_file, stchr)
    else:
        print(f"Directory '{input_global_dir}' does not exist or is not a directory.")

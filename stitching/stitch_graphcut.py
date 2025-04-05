import numpy as np
import cv2
import maxflow
import matplotlib.pyplot as plt
from pathlib import Path
from utils import _load_images, _load_transforms, _save, _warp


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


def find_overlap_region(mask1, mask2, eps=200):
    h, w = mask1.shape
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


def scaled_graph_cut(img1, img2, sure1, sure2, scale=2):
    if not sure2.any():
        return np.ones(img1.shape[:2], dtype='float32')
    orig_size = np.array((img1.shape[1], img1.shape[0]))
    new_size = (orig_size / scale).astype(int)
    smaller_1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LANCZOS4)
    smaller_sure1 = cv2.resize(sure1.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
    smaller_sure2 = cv2.resize(sure2.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)

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


def coarse_to_fine_optimal_seam(img1, img2, mask1, mask2, small_in_wide_slice,
                                coarse_scale=8, fine_scale=2, lane_width=200):
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    if only2.sum() < 0.001 * only2.size:
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


def _warp_coarse_to_fine(images, transforms, panorama_size):
    n = len(images)
    pano, pano_mask = _warp(images[0], transforms[0], panorama_size)
    img_indexes = pano_mask.astype('int8') - np.ones_like(pano_mask, dtype='int8')

    for i in range(1, n):
        warped_img, warped_mask = _warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(pano_mask, warped_mask)

        inter_img1 = pano[wide_window_slice]
        inter_mask1 = pano_mask[wide_window_slice]
        inter_img2 = warped_img[wide_window_slice]
        inter_mask2 = warped_mask[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter_img1, inter_img2, inter_mask1, inter_mask2,
                                             small_in_wide_slice, coarse_scale=16, fine_scale=4, lane_width=200)

        warped_mask[wide_window_slice] = np.where(labels, False, True)
        pano = np.where(warped_mask[..., np.newaxis], warped_img, pano)
        img_indexes = np.where(warped_mask, i, img_indexes)
        pano_mask = warped_mask | pano_mask

    return pano, img_indexes


def stitch_graphcut(transforms_file, output_file):

    transforms, panorama_size, img_paths = _load_transforms(transforms_file)
    pics = _load_images(img_paths)
    pano, img_indexes = _warp_coarse_to_fine(pics, transforms, panorama_size)

    img_indexes_name = output_file.name
    img_indexes_dir = output_file.parent / Path("img_indexes")
    img_indexes_dir.mkdir(parents=True, exist_ok=True)
    img_indexes_file = img_indexes_dir / img_indexes_name

    # Создаем фигуру и оси
    plt.figure(figsize=(15, 10))
    plt.imshow(img_indexes, cmap='viridis')
    plt.colorbar(label='Значения индексов')
    plt.axis('off')
    plt.savefig(img_indexes_file, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

    _save(pano, output_file)

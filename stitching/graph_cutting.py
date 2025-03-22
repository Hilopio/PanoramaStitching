import numpy as np
import cv2
import maxflow

borderValue = 1

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


def warp_coarse_to_fine(images, transforms, panorama_size):
    n = len(images)
    panorama_ans = warp(images[0], transforms[0], panorama_size)

    for i in range(1, n):
        warped_pic = warp(images[i], transforms[i], panorama_size)

        small_window_slice, wide_window_slice, small_in_wide_slice = find_overlap_region(panorama_ans, warped_pic)

        inter1 = panorama_ans[wide_window_slice]
        inter2 = warped_pic[wide_window_slice]

        labels = coarse_to_fine_optimal_seam(inter1, inter2, small_in_wide_slice,
                                             coarse_scale=16, fine_scale=4, lane_width=200)

        mask1 = np.any(panorama_ans != borderValue, axis=2)
        panorama_ans = np.where(mask1[..., np.newaxis], panorama_ans, warped_pic)
        panorama_ans[wide_window_slice] = np.where(labels[..., np.newaxis], inter1, inter2)

    return panorama_ans
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Union


def transform_and_stitch(transforms: List[np.ndarray], directory: Union[str, os.PathLike]) -> np.ndarray:
    """stitches all the images in the directory 
    into a panorama using known transformations 

    Args:
        transforms : list of 3 x 3 homographyies
        directory : path to the directory with the panorama parts

    Returns:
        panorama
    """
    pic_names = os.listdir(directory)
    n = len(pic_names)
    pics = []
    for name in pic_names:
        img = cv2.imread(os.path.join(directory, name)).astype(np.float32)
        pics.append(img)

    all_corners = np.empty((n, 4, 3))
    for i in range(n):
        all_corners[i] = [[0, 0, 1], [pics[i].shape[1], 0, 1], [pics[i].shape[1], pics[i].shape[0], 1], [0, pics[i].shape[0], 1]]

    all_new_corners = np.empty((n, 4, 3))
    for i in range(n):
        all_new_corners[i] = [np.dot(transforms[i], corner) for corner in all_corners[i]]
    
    all_new_corners = all_new_corners.reshape(-3, 3)
    x_news = all_new_corners[:, 0] / all_new_corners[:, 2]
    y_news = all_new_corners[:, 1] / all_new_corners[:, 2]
        
    y_min = min(y_news)
    x_min = min(x_news)
    y_max = int(round(max(y_news)))
    x_max = int(round(max(x_news)))

    x_shift = -min(x_min, 0)
    y_shift = -min(y_min, 0)
    T = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]], dtype='float32')

    x_min = int(round(x_min))
    y_min = int(round(y_min))

    height_new = y_max - y_min 
    width_new = x_max - x_min
    size = (width_new, height_new)

    panorama_ans = cv2.warpPerspective(src=pics[0], M=T @ transforms[0], dsize=size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
    for i in range(1, n):
        cv2.warpPerspective(pics[i], T @ transforms[i], size, panorama_ans,
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT)
    return panorama_ans

def transform_from_vec(vec: np.ndarray, i: int, pivot: int) -> np.ndarray:
    """unflatten list into list of 3x3 homographyies

    Args:
        vec : all homographies (except pivot one, which is identical) flattened in vector
        i : the number of the homography that needs to be extracted
        pivot : the number of the pivot image

    Returns:
        homography of the i-th image
    """
    if i == pivot:
        return np.eye(3)
    elif i > pivot:
        i -= 1
    H = vec[8 * i : 8 * (i + 1)]
    H = np.array(  [[H[0], H[1], H[2]],
                    [H[3], H[4], H[5]],
                    [H[6], H[7],   1 ]])
    return H

def vec_from_transforms(Hs: List[np.ndarray], pivot: int) -> List[float]:
    """flatten list of 3x3 homographies into list

    Args:
        Hs : a list of all homographies
        pivot : the number of the pivot image

    Returns:
        all homographies (except pivot one, which is identical) flattened in vector
    """
    n = len(Hs)
    vec = np.empty(8 * (n - 1))
    for i in range(n):
        if i == pivot:
            continue
        elif i < pivot:
            H = Hs[i].reshape(-1)
            H = H[:-1]
            vec[8 * i : 8 * (i + 1)] = H
        else:
            H = Hs[i].reshape(-1)
            H = H[:-1]
            vec[8 * (i - 1) : 8 * i] = H
    return vec

def fun(X: List[float], inliers: List[np.ndarray], pivot: int) -> np.ndarray:
    """distances between the coordinates of all pairs of inliers in the final coordinates

    Args:
        X : all homographies (except pivot one, which is identical) flattened in vector
        inliers : list of inliers, each inlier is a np.ndarry((i, j, x, y, xx, yy))
            where i and j are the indices of the images corresponding to the inlier,
            (x, y) are the coordinates of the point on image i,
            and (xx, yy) are the coordinates of the point on image j
        pivot : the number of the pivot image

    Returns:
        a vector of distances between the coordinates of all pairs of inliers in the final coordinates 
    """
    output = []
    for i, j, x, y, xx, yy in inliers:
        Hi = transform_from_vec(X, i, pivot)
        Hj = transform_from_vec(X, j, pivot)

        first = np.dot(Hi,[x, y, 1])
        first /= first[2]
        second = np.dot(Hj, [xx, yy, 1])
        second /= second[2]
        output.append(first[0] - second[0])
        output.append(first[1] - second[1])

    return np.array(output)

def optimization(Hs: List[np.ndarray], inliers: List[np.ndarray], pivot: int) -> List[np.ndarray]:
    """global alignment using all inliers by adjusting all homography matrice

    Args:
        Hs : a list of all homographies
        inliers : list of inliers, each inlier is a np.ndarry((i, j, x, y, xx, yy))
            where i and j are the indices of the images corresponding to the inlier,
            (x, y) are the coordinates of the point on image i,
            and (xx, yy) are the coordinates of the point on image j
        pivot : the number of the pivot image

    Returns:
        a list of new homographies
    """
    n = len(Hs)
    vec = vec_from_transforms(Hs, pivot)
    norm = fun(vec, inliers, pivot)

    init_error = (norm ** 2).mean() ** 0.5 
    res_lm = least_squares(fun, vec, method='lm', xtol=1e-6, ftol=1e-6, args=(inliers, pivot))
    optim_error = (res_lm.fun ** 2).mean() ** 0.5
    new_vec = res_lm.x

    final_transforms = []
    for i in range(n):
        final_transforms.append(transform_from_vec(new_vec, i, pivot))
    return final_transforms, init_error, optim_error
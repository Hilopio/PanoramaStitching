import os
import cv2
import numpy as np
# import maxflow
from scipy.ndimage import shift
from scipy.optimize import least_squares



def vector_stitching(img1, img2):
    mask = img1 == -1
    img1[mask] = img2[mask]
    return img1

def transform_and_stitch(transforms, directory):
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
        warped_pic = cv2.warpPerspective(pics[i], T @ transforms[i], size, panorama_ans,
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT)
        # panorama_ans = seam_cut(panorama_ans, warped_pic)
        # panorama_ans = vector_stitching(panorama_ans, warped_pic)
    return panorama_ans

# def seam_cut(img1, img2):
#     diff = lambda img1, img2 : np.abs(img1 - img2).sum(axis=2)
#     def grad(image):
#         b, g, r = cv2.split(image)
#         b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
#         g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
#         r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
#         return b_energy + g_energy + r_energy
    
#     ans = vector_stitching(img1, img2)
#     intersection = (img1 * img2).sum(axis=2) > 0

    
#     inter_idx = np.nonzero(intersection)

#     if len(inter_idx[0]) < 5:
#         return ans
    
#     y_min = np.min(inter_idx[0])
#     y_max = np.max(inter_idx[0])
#     x_min = np.min(inter_idx[1])
#     x_max = np.max(inter_idx[1])

    
#     size = img1.shape[:2]
#     eps = 50
#     Y_MIN, Y_MAX, X_MIN, X_MAX = max(y_min - eps, 0), min(y_max + eps, size[0]), max(x_min - eps, 0), min(x_max + eps, size[1])
#     only_left = img1.sum(axis=2) * np.logical_not(intersection) > 0
#     only_right = img2.sum(axis=2) * np.logical_not(intersection) > 0
#     left_mask = only_left[Y_MIN:Y_MAX, X_MIN:X_MAX]
#     right_mask = only_right[Y_MIN:Y_MAX, X_MIN:X_MAX]
#     inter_mask = intersection[Y_MIN:Y_MAX, X_MIN:X_MAX]
#     inter_1 = img1[Y_MIN:Y_MAX, X_MIN:X_MAX]
#     inter_2 = img2[Y_MIN:Y_MAX, X_MIN:X_MAX]
#     scale = 2
#     size = (inter_1.shape[0], inter_1.shape[1])
#     new_size = (inter_1.shape[0] // scale, inter_1.shape[1] // scale)


#     smaler_1 = cv2.resize(inter_1, new_size[::-1])
#     smaler_2 = cv2.resize(inter_2, new_size[::-1])

#     grad1 = grad(smaler_1)
#     grad2 = grad(smaler_2)
#     difference = diff(smaler_1, smaler_2)
#     grad_difference = np.abs(grad1 - grad2)
#     alpha = 1
#     smooth_map = difference + alpha * grad_difference

#     g = maxflow.Graph[float]()
#     nodeids = g.add_grid_nodes(new_size)
#     structure = np.array([[0, 0, 0],
#                             [0, 0, 1],
#                             [0, 0, 0],])

#     a = shift(smooth_map, (0, 1), mode='nearest') + smooth_map
#     g.add_grid_edges(nodeids, a, structure=structure, symmetric=True)

#     structure = np.array([[0, 1, 0],
#                             [0, 0, 0],
#                             [0, 0, 0],])
#     a = shift(smooth_map, (1, 0), mode='nearest') + smooth_map
#     g.add_grid_edges(nodeids, a, structure=structure, symmetric=True)

#     left_mask = cv2.resize(left_mask.astype(int), new_size[::-1], interpolation = cv2.INTER_NEAREST)
#     right_mask = cv2.resize(right_mask.astype(int), new_size[::-1], interpolation = cv2.INTER_NEAREST)
    
#     left_inf = []
#     right_inf = []

#     for i in range(nodeids.shape[0]):
#         for j in range(nodeids.shape[1]):
#             if left_mask[i][j]:
#                 left_inf.append(nodeids[i][j])
#             if right_mask[i][j]:
#                 right_inf.append(nodeids[i][j])

#     g.add_grid_tedges(np.array(left_inf), np.inf, 0)
#     g.add_grid_tedges(np.array(right_inf), 0, np.inf)

#     g.maxflow()
#     sgm = g.get_grid_segments(nodeids)
#     lbls_mask = np.int_(np.logical_not(sgm))
#     lbls_mask = lbls_mask.astype('float32')
#     lbls_mask = cv2.resize(lbls_mask, size[::-1], interpolation = cv2.INTER_NEAREST)


#     lbls_mask = np.stack((lbls_mask, lbls_mask, lbls_mask), axis=2)
#     gcans = inter_1 * lbls_mask + inter_2 * (1 - lbls_mask)

#     newans = ans.copy()
#     intersection3 = np.stack((intersection, intersection, intersection), axis=-1)
#     newans *= np.logical_not(intersection3)

#     newans[Y_MIN:Y_MAX, X_MIN:X_MAX] += gcans * np.stack((inter_mask, inter_mask, inter_mask), axis=-1)
#     return newans

def transform_from_vec(vec, i, pivot):
    if i == pivot:
        return np.eye(3)
    elif i > pivot:
        i -= 1
    H = vec[8 * i : 8 * (i + 1)]
    H = np.array(  [[H[0], H[1], H[2]],
                    [H[3], H[4], H[5]],
                    [H[6], H[7],   1 ]])
    return H

def vec_from_transforms(Hs, pivot):
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

def fun(X, inliers, pivot):
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

def optimization(Hs, inliers, pivot):
    n = len(Hs)
    vec = vec_from_transforms(Hs, pivot)
    norm = fun(vec, inliers, pivot)

    init_error = (norm ** 2).mean() ** 0.5 
    # print(f'initial errors = {init_error:.4}')

    res_lm = least_squares(fun, vec, method='lm', xtol=1e-6, ftol=1e-6, args=(inliers, pivot))
    optim_error = (res_lm.fun ** 2).mean() ** 0.5
    # print(f'optimized errors = {optim_error:.4}')
    new_vec = res_lm.x
    final_transforms = []
    for i in range(n):
        final_transforms.append(transform_from_vec(new_vec, i, pivot))
    return final_transforms, init_error, optim_error
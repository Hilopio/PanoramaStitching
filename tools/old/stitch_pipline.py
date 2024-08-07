import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [8, 8]
from stitch_functions import *
from PIL import Image
from scipy.optimize import least_squares
import pickle
import maxflow
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import shift


postfix = ''

postfix = '_ShiftN'
postfix = '_undistorced'



def find_all_homographies(panorama_size, refer_unit, directory):
    hhs = np.empty((panorama_size[0], panorama_size[1] - 1, 3, 3))
    vhs = np.empty((panorama_size[0] - 1, panorama_size[1], 3, 3))
    hinliers = []
    vinliers = []
    #horizontal homographies
    for i in range(panorama_size[0]):
        for j in range(panorama_size[1] - 1):
            if j < refer_unit[1] - 1:
                inliers, hhs[i,j] = find_homo(f'{directory}/{i+1}.{j+1}{postfix}.jpg', f'{directory}/{i+1}.{j+2}{postfix}.jpg')  
                hinliers.append(inliers)
                #print(inliers.shape)
            else:
                inliers, hhs[i,j] = find_homo(f'{directory}/{i+1}.{j+2}{postfix}.jpg', f'{directory}/{i+1}.{j+1}{postfix}.jpg')
                hinliers.append(inliers)
                #print(inliers.shape)


    #vertical homographies
    for i in range(panorama_size[0] - 1):
        for j in range(panorama_size[1]):
            if i < refer_unit[0] - 1:
                inliers, vhs[i,j] = find_homo(f'{directory}/{i+1}.{j+1}{postfix}.jpg', f'{directory}/{i+2}.{j+1}{postfix}.jpg')
                vinliers.append(inliers)
                #print(inliers.shape)
            else:
                inliers, vhs[i,j] = find_homo(f'{directory}/{i+2}.{j+1}{postfix}.jpg', f'{directory}/{i+1}.{j+1}{postfix}.jpg')
                vinliers.append(inliers)
                #print(inliers.shape)

    return hhs, vhs, vinliers, hinliers

def get_all_transforms(panorama_size, refer_unit, hhs, vhs):
    transforms = np.empty((panorama_size[0], panorama_size[1], 3, 3))
    for i in range(panorama_size[0]):
        vH = np.eye(3)
        if i < refer_unit[0] - 1:
            for ii in range(i, refer_unit[0] - 1):
                vH = vhs[ii, refer_unit[1] - 1] @ vH
        elif i == refer_unit[0] - 1:
            pass
        else:
            for ii in range(i, refer_unit[0] - 1, -1):
                vH = vhs[ii-1, refer_unit[1] - 1] @ vH
        transforms[i, refer_unit[1] - 1] = vH
        for j in range(panorama_size[1]):
            hH = np.eye(3)
            if j == refer_unit[1] - 1:
                pass
            elif j < refer_unit[1] - 1:
                for jj in range(j, refer_unit[1] - 1):
                    hH = hhs[i, jj] @ hH
            else:
                for jj in range(j, refer_unit[1] - 1, -1):
                    hH = hhs[i, jj-1] @ hH
            transforms[i, j] = vH @ hH

    return transforms

def get_averaged_transforms(panorama_size, refer_unit, hhs, vhs):
    from sympy.utilities.iterables import multiset_permutations
    transforms = np.empty((panorama_size[0], panorama_size[1], 3, 3))

    for i in range(panorama_size[0]):
        for j in range(panorama_size[1]):
            if (i + 1, j + 1) == refer_unit:
                transforms[i, j] = np.eye(3)
                continue
            vertical = np.zeros(np.abs(refer_unit[0] - 1 - i), dtype=int)
            horiz = np.ones(np.abs(refer_unit[1] - 1 - j), dtype=int)
            way_mask = np.concatenate((vertical, horiz), axis=0)
            all_ways = np.array(list(multiset_permutations(way_mask)))
            ways_num = all_ways.shape[0]
            ways_len = all_ways.shape[1]
            ways = list()
            for k in range(ways_num):
                H = np.eye(3)
                ii = i
                jj = j
                for r in range(ways_len):
                    if all_ways[k, r] == 0:
                        if ii < refer_unit[0] - 1:
                            H = vhs[ii, jj] @ H
                            ii += 1
                        elif ii > refer_unit[0] - 1:
                            H = vhs[ii - 1, jj] @ H
                            ii -= 1
                        else:
                            raise PermissionError('sth wrong')
                    else:
                        if jj < refer_unit[1] - 1:
                            H = hhs[ii, jj] @ H
                            jj += 1
                        elif jj > refer_unit[1] - 1:
                            H = hhs[ii, jj - 1] @ H
                            jj -= 1
                        else:
                            raise PermissionError('sth wrong')
                ways.append(H)
            ways = np.array(ways)
            #avrged_H = np.mean(ways, axis=0)
            avrged_H = np.median(ways, axis=0)
            assert avrged_H.shape == (3, 3)
            avrged_H /= avrged_H[2, 2]
            transforms[i, j] = avrged_H

    return transforms

def transform_from_vec(panorama_size, refer_unit, vec, i, j):
    if (i + 1, j + 1) == refer_unit:
        return np.eye(3)
    if (i * panorama_size[1] + j < (refer_unit[0] - 1) * panorama_size[1] + refer_unit[1] - 1):
        h = vec[8 * (i * panorama_size[1] + j) : 8 * (i * panorama_size[1] + j + 1)]
    else:
        h = vec[8 * (i * panorama_size[1] + j - 1) : 8 * (i * panorama_size[1] + j)]
    H = np.array(  [[h[0], h[1], h[2]],
                    [h[3], h[4], h[5]],
                    [h[6], h[7],   1 ]])
    return H

def vec_from_transforms(panorama_size, refer_unit, transforms):
    vec = np.empty((8 * (panorama_size[0] * panorama_size[1] - 1)))
    for i in range(panorama_size[0]):
        for j in range(panorama_size[1]):
            if (i + 1, j + 1) == refer_unit:
                continue
            h = transforms[i,j].reshape(-1)
            h = h[:-1]
            if (i * panorama_size[1] + j < (refer_unit[0] - 1) * panorama_size[1] + refer_unit[1] - 1):
                vec[8 * (i * panorama_size[1] + j) : 8 * (i * panorama_size[1] + j + 1)] = h
            else:
                vec[8 * (i * panorama_size[1] + j - 1) : 8 * (i * panorama_size[1] + j)] = h
    return vec

def fun(X, panorama_size, refer_unit, vinliers, hinliers):
    output = []
    #horizontal inliers
    for i in range(panorama_size[0]):
        for j in range(panorama_size[1] - 1):
            s = i * (panorama_size[1] - 1) + j
            inliers = hinliers[s]
            H_left = transform_from_vec(panorama_size, refer_unit, X, i, j)
            H_right = transform_from_vec(panorama_size, refer_unit, X, i, j+1)
            left = np.ones((inliers.shape[0], 3))
            right = np.ones((inliers.shape[0], 3))
            if j < refer_unit[1] - 1:
                left[:, :2] = inliers[:, :2]
                right[:, :2] = inliers[:, 2:]
            else:
                left[:, :2] = inliers[:, 2:]
                right[:, :2] = inliers[:, :2]
            left_ims = np.array([np.dot(H_left, point) for point in left])
            right_ims = np.array([np.dot(H_right, point) for point in right])
            left_ims = np.stack((left_ims[:, 0] / left_ims[:, 2], left_ims[:, 1] / left_ims[:, 2],)).T
            right_ims = np.stack((right_ims[:, 0] / right_ims[:, 2], right_ims[:, 1] / right_ims[:, 2],)).T
            for k in range(left_ims.shape[0]):
                output.append(left_ims[k][0] - right_ims[k][0])
                output.append(left_ims[k][1] - right_ims[k][1])
    
    #vertical inliers
    for i in range(panorama_size[0] - 1):
        for j in range(panorama_size[1]):
            s = i * (panorama_size[1]) + j
            inliers = vinliers[s]
            H_upper = transform_from_vec(panorama_size, refer_unit, X, i, j)
            H_lower = transform_from_vec(panorama_size, refer_unit, X, i+1, j)
            upper = np.ones((inliers.shape[0], 3))
            lower = np.ones((inliers.shape[0], 3))
            if i < refer_unit[0] - 1:
                upper[:, :2] = inliers[:, :2]
                lower[:, :2] = inliers[:, 2:]
            else:
                upper[:, :2] = inliers[:, 2:]
                lower[:, :2] = inliers[:, :2]
            upper_ims = np.array([np.dot(H_upper, point) for point in upper])
            lower_ims = np.array([np.dot(H_lower, point) for point in lower])
            upper_ims = np.stack((upper_ims[:, 0] / upper_ims[:, 2], upper_ims[:, 1] / upper_ims[:, 2],)).T
            lower_ims = np.stack((lower_ims[:, 0] / lower_ims[:, 2], lower_ims[:, 1] / lower_ims[:, 2],)).T
            for k in range(upper_ims.shape[0]):
                output.append(upper_ims[k][0] - lower_ims[k][0])
                output.append(upper_ims[k][1] - lower_ims[k][1])

    return np.array(output)

def optimization(transforms, panorama_size, refer_unit,  vinliers, hinliers):
    vec = vec_from_transforms(panorama_size, refer_unit, transforms)
    norm = fun(vec, panorama_size, refer_unit, vinliers, hinliers)
    print("initial errors = ", (norm ** 2).mean() ** 0.5)

    res_lm = least_squares(fun, vec, method='lm',  args=(panorama_size, refer_unit, vinliers, hinliers), xtol=1e-15, ftol=1e-15)
    # print("\nmethod = lm")
    # print("cost = ", res_lm.cost)
    # print("fun = ", res_lm.fun)
    rmse = (res_lm.fun ** 2).mean() ** 0.5
    print("optimized errors = ", rmse)
    print(res_lm.status)
    # res_trf = least_squares(fun, vec, method='trf', args=(panorama_size, refer_unit, vinliers, hinliers)) # bounds=(lb, ub),
    # print("\nmethod = trf")
    # print("cost = ", res_trf.cost)
    # print("fun = ", res_trf.fun)
    # print("optimized errors = ", (res_trf.fun ** 2).mean() ** 0.5)

    new_vec = res_lm.x
    new_transforms = np.empty(transforms.shape)
    for i in range(panorama_size[0]):
        for j in range(panorama_size[1]):
            new_transforms[i,j] = transform_from_vec(panorama_size, refer_unit, new_vec, i, j)

    return new_transforms, rmse
    
    



def seam_cut(img1, img2, direction='h'):
    diff = lambda img1, img2 : np.abs(img1 - img2).sum(axis=2)

    def grad(image):
        b, g, r = cv2.split(image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

    ans = vector_stitching(img1, img2)
    intersection = np.sign(img1 * img2)
    indexes = np.nonzero(intersection)
    y_min = np.min(indexes[0])
    y_max = np.max(indexes[0])
    x_min = np.min(indexes[1])
    x_max = np.max(indexes[1])
    inter_1 = img1[y_min:y_max, x_min:x_max]
    inter_2 = img2[y_min:y_max, x_min:x_max]
    inter_mask = intersection[y_min:y_max, x_min:x_max]


    scale = 2
    size = (inter_1.shape[0], inter_1.shape[1])
    new_size = (inter_1.shape[0] // scale, inter_1.shape[1] // scale)
    smaler_1 = cv2.resize(inter_1, new_size[::-1])
    smaler_2 = cv2.resize(inter_2, new_size[::-1])

    grad1 = grad(smaler_1)
    grad2 = grad(smaler_2 )
    difference = diff(smaler_1, smaler_2)
    grad_difference = np.abs(grad1 - grad2)
    alpha = 1
    smooth_map = difference + alpha * grad_difference

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(new_size)
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0],])

    a = shift(smooth_map, (0, 1), mode='nearest') + smooth_map
    g.add_grid_edges(nodeids, a, structure=structure, symmetric=True)

    structure = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0],])
    a = shift(smooth_map, (1, 0), mode='nearest') + smooth_map
    g.add_grid_edges(nodeids, a, structure=structure, symmetric=True)

    
    if direction == 'h':
        height = nodeids.shape[0]
        # Source node connected to leftmost non-terminal nodes.
        g.add_grid_tedges(nodeids[:, 0], np.inf, 0)
        # Sink node connected to rightmost non-terminal nodes.
        g.add_grid_tedges(nodeids[:, -1], 0, np.inf)
    else:
        width = nodeids.shape[1]
        # Source node connected to topmost non-terminal nodes.
        g.add_grid_tedges(nodeids[0], np.inf, 0)
        # Sink node connected to rightmost non-terminal nodes.
        g.add_grid_tedges(nodeids[-1], 0, np.inf)

    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    lbls_mask = np.int_(np.logical_not(sgm))
    lbls_mask = lbls_mask.astype('float32')
    lbls_mask = cv2.resize(lbls_mask, size[::-1], interpolation = cv2.INTER_NEAREST)

    # plt.imshow(lbls_mask, cmap=plt.cm.gray, interpolation='nearest')
    # plt.show()
    lbls_mask = np.stack((lbls_mask, lbls_mask, lbls_mask), axis=2)
    gcans = inter_1 * lbls_mask + inter_2 * (1 - lbls_mask)

    newans = ans.copy()
    newans *= (1 - intersection)

    newans[y_min:y_max, x_min:x_max] += gcans * inter_mask
    return newans

def transform_and_stitch(panorama_size, refer_unit, directory, transforms):
    pics = list()
    for i in range(panorama_size[0]):
        for j in range (panorama_size[1]):
            pic = cv2.imread(f'{directory}/{i+1}.{j+1}{postfix}.jpg')
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            pics.append(pic)


    n = panorama_size[0] * panorama_size[1]
    all_corners = np.empty((n, 4, 3))
    for i in range(n):
        all_corners[i] = [[0, 0, 1], [pics[i].shape[1], 0, 1], [pics[i].shape[1], pics[i].shape[0], 1], [0, pics[i].shape[0], 1]]

    all_new_corners = np.empty((n, 4, 3))
    for i in range(n):
        s = i // panorama_size[1]
        c = i % panorama_size[1]
        all_new_corners[i] = [np.dot(transforms[s,c], corner) for corner in all_corners[i]]

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

    panorama_ans = np.zeros((height_new, width_new, 3))
    warped_pic = cv2.warpPerspective(src=pics[0], M=T @ transforms[0, 0], dsize=size, flags=cv2.INTER_NEAREST)
    panorama_ans += warped_pic
    for column in tqdm(range(1, panorama_size[1])):
        warped_pic = cv2.warpPerspective(src=pics[column], M=T @ transforms[0, column], dsize=size, flags=cv2.INTER_NEAREST)
        #panorama_ans = vector_stitching(panorama_ans, warped_pic)
        panorama_ans = seam_cut(panorama_ans, warped_pic)

    for row in tqdm(range(1, panorama_size[0])):
        row_ans = np.zeros((height_new, width_new, 3))
        warped_pic = cv2.warpPerspective(src=pics[row * panorama_size[1]], M=T @ transforms[row, 0], dsize=size, flags=cv2.INTER_NEAREST)
        row_ans = vector_stitching(row_ans, warped_pic)
        for column in tqdm(range(1, panorama_size[1])):
            warped_pic = cv2.warpPerspective(src=pics[row * panorama_size[1] + column], M=T @ transforms[row, column], dsize=size,
                                            flags=cv2.INTER_NEAREST)
            #row_ans = vector_stitching(row_ans, warped_pic)
            row_ans = seam_cut(row_ans, warped_pic)

        # panorama_ans = vector_stitching(panorama_ans, row_ans)
        panorama_ans = seam_cut(panorama_ans, row_ans, 'v')
    return panorama_ans

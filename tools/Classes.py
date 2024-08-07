import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [8, 8]
from stitch_functions import *
from scipy.optimize import least_squares
import maxflow
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from sympy.utilities.iterables import multiset_permutations


postfix = ''
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

class Panorama():
    def __init__(self, dir, size=None, pivot=None, output=None):
        self.dir = '../dataset/' + dir

        if size is None:
            ls = os.listdir(self.dir)
            if len(ls) == 20:
                self.size = (4, 5)
            elif len(ls) == 25:
                self.size = (5, 5)
            else:
                raise 'remove all extra files or specify panorama size'
        else:
            self.size = size

        if pivot is None:
            self.pivot = (self.size[0] // 2 + 1, self.size[1] // 2 + 1)
        else:
            self.pivot = pivot

        if output is None:
            self.output = '../dataset/' + dir + '_pano.jpg'
        else:
            self.output = output
        
    def info(self):
        print(f'dir = {self.dir}')
        print(f'size = {self.size}')
        print(f'pivot = {self.pivot}')
        print(f'output = {self.output}')

    def get_all_homographies(self):
        self.hhs = np.empty((self.size[0], self.size[1] - 1, 3, 3))
        self.vhs = np.empty((self.size[0] - 1, self.size[1], 3, 3))
        self.hinliers = []
        self.vinliers = []

        #horizontal homographies
        for i in range(self.size[0]):
            for j in range(self.size[1] - 1):
                if j < self.pivot[1] - 1:
                    inliers, self.hhs[i,j] = find_homo(f'{self.dir}/{i+1}.{j+1}{postfix}.jpg', f'{self.dir}/{i+1}.{j+2}{postfix}.jpg')  
                    self.hinliers.append(inliers)
                else:
                    inliers, self.hhs[i,j] = find_homo(f'{self.dir}/{i+1}.{j+2}{postfix}.jpg', f'{self.dir}/{i+1}.{j+1}{postfix}.jpg')
                    self.hinliers.append(inliers)


        #vertical homographies
        for i in range(self.size[0] - 1):
            for j in range(self.size[1]):
                if i < self.pivot[0] - 1:
                    inliers, self.vhs[i,j] = find_homo(f'{self.dir}/{i+1}.{j+1}{postfix}.jpg', f'{self.dir}/{i+2}.{j+1}{postfix}.jpg')
                    self.vinliers.append(inliers)
                else:
                    inliers, self.vhs[i,j] = find_homo(f'{self.dir}/{i+2}.{j+1}{postfix}.jpg', f'{self.dir}/{i+1}.{j+1}{postfix}.jpg')
                    self.vinliers.append(inliers)



    def get_all_transforms(self):
        self.transforms = np.empty((self.size[0], self.size[1], 3, 3))
        for i in range(self.size[0]):
            vH = np.eye(3)
            if i < self.pivot[0] - 1:
                for ii in range(i, self.pivot[0] - 1):
                    vH = self.vhs[ii, self.pivot[1] - 1] @ vH
            elif i == self.pivot[0] - 1:
                pass
            else:
                for ii in range(i, self.pivot[0] - 1, -1):
                    vH = self.vhs[ii-1, self.pivot[1] - 1] @ vH
            self.transforms[i, self.pivot[1] - 1] = vH
            for j in range(self.size[1]):
                hH = np.eye(3)
                if j == self.pivot[1] - 1:
                    pass
                elif j < self.pivot[1] - 1:
                    for jj in range(j, self.pivot[1] - 1):
                        hH = self.hhs[i, jj] @ hH
                else:
                    for jj in range(j, self.pivot[1] - 1, -1):
                        hH = self.hhs[i, jj-1] @ hH
                self.transforms[i, j] = vH @ hH

    def get_averaged_transforms(self):
        self.transforms = np.empty((self.size[0], self.size[1], 3, 3))

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i + 1, j + 1) == self.pivot:
                    self.transforms[i, j] = np.eye(3)
                    continue
                vertical = np.zeros(np.abs(self.pivot[0] - 1 - i), dtype=int)
                horiz = np.ones(np.abs(self.pivot[1] - 1 - j), dtype=int)
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
                            if ii < self.pivot[0] - 1:
                                H = self.vhs[ii, jj] @ H
                                ii += 1
                            elif ii > self.pivot[0] - 1:
                                H = self.vhs[ii - 1, jj] @ H
                                ii -= 1
                            else:
                                raise PermissionError('sth wrong')
                        else:
                            if jj < self.pivot[1] - 1:
                                H = self.hhs[ii, jj] @ H
                                jj += 1
                            elif jj > self.pivot[1] - 1:
                                H = self.hhs[ii, jj - 1] @ H
                                jj -= 1
                            else:
                                raise PermissionError('sth wrong')
                    ways.append(H)
                ways = np.array(ways)
                #avrged_H = np.mean(ways, axis=0)
                avrged_H = np.median(ways, axis=0)
                assert avrged_H.shape == (3, 3)
                avrged_H /= avrged_H[2, 2]
                self.transforms[i, j] = avrged_H


    def transform_from_vec(self, vec, i, j):
        if (i + 1, j + 1) == self.pivot:
            return np.eye(3)
        if (i * self.size[1] + j < (self.pivot[0] - 1) * self.size[1] + self.pivot[1] - 1):
            h = vec[8 * (i * self.size[1] + j) : 8 * (i * self.size[1] + j + 1)]
        else:
            h = vec[8 * (i * self.size[1] + j - 1) : 8 * (i * self.size[1] + j)]
        H = np.array(  [[h[0], h[1], h[2]],
                        [h[3], h[4], h[5]],
                        [h[6], h[7],   1 ]])
        return H

    def vec_from_transforms(self):
        vec = np.empty((8 * (self.size[0] * self.size[1] - 1)))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i + 1, j + 1) == self.pivot:
                    continue
                h = self.transforms[i,j].reshape(-1)
                h = h[:-1]
                if (i * self.size[1] + j < (self.pivot[0] - 1) * self.size[1] + self.pivot[1] - 1):
                    vec[8 * (i * self.size[1] + j) : 8 * (i * self.size[1] + j + 1)] = h
                else:
                    vec[8 * (i * self.size[1] + j - 1) : 8 * (i * self.size[1] + j)] = h
        return vec

    def fun(self, X):
        output = []
        #horizontal inliers
        for i in range(self.size[0]):
            for j in range(self.size[1] - 1):
                s = i * (self.size[1] - 1) + j
                inliers = self.hinliers[s]
                H_left = self.transform_from_vec(X, i, j)
                H_right = self.transform_from_vec(X, i, j+1)
                left = np.ones((inliers.shape[0], 3))
                right = np.ones((inliers.shape[0], 3))
                if j < self.pivot[1] - 1:
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
        for i in range(self.size[0] - 1):
            for j in range(self.size[1]):
                s = i * (self.size[1]) + j
                inliers = self.vinliers[s]
                H_upper = self.transform_from_vec(X, i, j)
                H_lower = self.transform_from_vec(X, i+1, j)
                upper = np.ones((inliers.shape[0], 3))
                lower = np.ones((inliers.shape[0], 3))
                if i < self.pivot[0] - 1:
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



    def optimization(self):
        vec = self.vec_from_transforms()
        norm = self.fun(vec)

        print("initial errors = ", (norm ** 2).mean() ** 0.5)

        res_lm = least_squares(self.fun, vec, method='lm', xtol=1e-15, ftol=1e-15)
        self.rmse = (res_lm.fun ** 2).mean() ** 0.5
        print("optimized errors = ", self.rmse)
        new_vec = res_lm.x
        self.final_transforms = np.empty(self.transforms.shape)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.final_transforms[i,j] = self.transform_from_vec(new_vec, i, j)

    def transform_and_stitch(self):
        pics = list()
        for i in range(self.size[0]):
            for j in range (self.size[1]):
                pic = cv2.imread(f'{self.dir}/{i+1}.{j+1}{postfix}.jpg').astype(np.float32)
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                pics.append(pic)


        n = self.size[0] * self.size[1]
        all_corners = np.empty((n, 4, 3))
        for i in range(n):
            all_corners[i] = [[0, 0, 1], [pics[i].shape[1], 0, 1], [pics[i].shape[1], pics[i].shape[0], 1], [0, pics[i].shape[0], 1]]

        all_new_corners = np.empty((n, 4, 3))
        for i in range(n):
            s = i // self.size[1]
            c = i % self.size[1]
            all_new_corners[i] = [np.dot(self.transforms[s,c], corner) for corner in all_corners[i]]

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

        self.panorama_ans = np.zeros((height_new, width_new, 3))
        warped_pic = cv2.warpPerspective(src=pics[0], M=T @ self.transforms[0, 0], dsize=size, flags=cv2.INTER_NEAREST)
        self.panorama_ans += warped_pic
        for column in tqdm(range(1, self.size[1])):
            warped_pic = cv2.warpPerspective(src=pics[column], M=T @ self.transforms[0, column], dsize=size, flags=cv2.INTER_NEAREST)
            #panorama_ans = vector_stitching(panorama_ans, warped_pic)
            self.panorama_ans = seam_cut(self.panorama_ans, warped_pic)

        for row in tqdm(range(1, self.size[0])):
            row_ans = np.zeros((height_new, width_new, 3))
            warped_pic = cv2.warpPerspective(src=pics[row * self.size[1]], M=T @ self.transforms[row, 0], dsize=size, flags=cv2.INTER_NEAREST)
            row_ans = vector_stitching(row_ans, warped_pic)
            for column in tqdm(range(1, self.size[1])):
                warped_pic = cv2.warpPerspective(src=pics[row * self.size[1] + column], M=T @ self.transforms[row, column], dsize=size,
                                                flags=cv2.INTER_NEAREST)
                #row_ans = vector_stitching(row_ans, warped_pic)
                row_ans = seam_cut(row_ans, warped_pic)

            # panorama_ans = vector_stitching(panorama_ans, row_ans)
            self.panorama_ans = seam_cut(self.panorama_ans, row_ans, 'v')

    def save_result(self):
        final_pic = self.panorama_ans.astype('uint8') 
        final_bgr = cv2.cvtColor(final_pic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.output, final_bgr)

    def pipline(self):    
        start_time = time.time()
        
        print("finding homographies...")
        self.get_all_homographies()

        print("transforming...")
        self.get_averaged_transforms()

        print("optimizing...")
        self.optimization()
      
        print("stitching...")
        self.transform_and_stitch()

        print("drawing...")
        self.save_result()
        
        total_time = time.time() - start_time
        print(f"completed! time spent: {round(total_time) // 60} min {total_time % 60} sec")
        print(f'rmse = {self.rmse}')
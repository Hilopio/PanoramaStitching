import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [8, 8]
from PIL import Image

def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT(img):
    # siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
    siftDetector= cv2.SIFT_create()

    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2) #bf.match(des1,des2)
    print(f"raw matches number = {len(matches)}")

    
    matches = list(matches)
    matches.sort(key=lambda match: np.abs(match[0].distance / match[1].distance))
    good = matches[:75]    
    # good = []
    # for m,n in matches:
    #     if m.distance < threshold*n.distance:
    #         good.append([m])
    matches = []
    for pair in good:
        # trainIdx - откуда
        # queryIdx - куда
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    print(f"good matches number = {len(matches)}")

    return matches

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type

    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')

    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    #print(H)
    return H

def homography_average(pairs):
    rows = []
    Z = np.ones((pairs.shape[0], 3))
    B1 = np.empty((pairs.shape[0]))
    B2 = np.empty((pairs.shape[0]))
    W1 = np.empty((pairs.shape[0], 2))
    W2 = np.empty((pairs.shape[0], 2))
    for i in range(pairs.shape[0]):
        B1[i] = pairs[i][2]
        B2[i] = pairs[i][3]
        Z[i, :2] = pairs[i][0:2]
        W1[i] = [-pairs[i][2]*pairs[i][0], -pairs[i][2]*pairs[i][1]]
        W2[i] = [-pairs[i][3]*pairs[i][0], -pairs[i][3]*pairs[i][1]]

    ATA = np.block([[Z.T @ Z, np.zeros((3, 3)), Z.T @ W1],
                    [np.zeros((3, 3)), Z.T @ Z, Z.T @ W2],
                    [W1.T @ Z, W2.T @ Z, W1.T @ W1 + W2.T @ W2]])
    ATB = np.block([Z.T @ B1, Z.T @ B2, W1.T @ B1 + W2.T @ B2]).T
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
            # print("oopse")
            # return np.zeros((3,3), dtype=int)
            pass
    try:
        hs = np.linalg.solve(ATA, ATB)
    except np.LinAlgError:
        return np.zeros((3,3), dtype=int)
    
    H = np.array([[hs[0], hs[1], hs[2]],
                  [hs[3], hs[4], hs[5]],
                  [hs[6], hs[7], 1.0]])
    #print(H)
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    # idx = random.choices(range(len(matches)), k)
    
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0

    for i in range(iters):
        points = random_point(matches, k=25)
        H = homography_average(points)
        #  avoid dividing by zero
        if np.linalg.matrix_rank(H) < 3:
            continue
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        # print(num_inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    final_H = homography_average(best_inliers)
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, final_H

def vector_stitching(img1, img2):
    size = img1.shape
    ans = np.copy(img1)
    intersection = np.sign((img1 * img2).sum(axis=2))
    intersection = np.broadcast_to(intersection, (3, size[0], size[1]))
    intersection = np.transpose(intersection, (1, 2, 0)) * (-1) + 1
    ans += img2*intersection
    return ans

def stitch_img(left, right, H):
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    height_l, width_l, channel_l = left.shape
    height_r, width_r, channel_r = right.shape


    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]

    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T

    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    y_max = int(round(max(y_news)))
    x_max = int(round(max(x_news)))
    
    
    x_shift = -min(x_min, 0)
    y_shift = -min(y_min, 0)
    translation_mat = np.array([[1, 0, x_shift], [0, 1, y_shift], [0, 0, 1]], dtype='float32')

    H = np.dot(translation_mat, H)

    x_min = int(round(x_min))
    y_min = int(round(y_min))
    x_shift = -min(x_min, 0)
    y_shift = -min(y_min, 0)


    # Get height, width
    height_new = max(height_r + y_shift, y_max) 
    width_new = max(width_r + x_shift, x_max)
    size = (width_new, height_new)
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size) 
    
    start_time = time.time()
    stitched = vector_stitching(warped_l, warped_r)
    end_time = time.time()
    #print(f"stitching time = {end_time - start_time}")
    return stitched

def elementary_stitching(img1, img2, output="temp.jpg"):
    left_gray, _, left_rgb = read_image(img1)
    right_gray, _, right_rgb = read_image(img2)

    cut_left_gray = cv2.resize(left_gray, (left_gray.shape[1]//2,left_gray.shape[0]//2))
    cut_right_gray = cv2.resize(right_gray, (right_gray.shape[1]//2,right_gray.shape[0]//2))
    cut_left_rgb = cv2.resize(left_rgb, (left_rgb.shape[1]//2,left_rgb.shape[0]//2))
    cut_right_rgb = cv2.resize(right_rgb, (right_rgb.shape[1]//2,right_rgb.shape[0]//2))
    
    kp_left, des_left = SIFT(cut_left_gray)
    kp_right, des_right = SIFT(cut_right_gray)

    matches = matcher(kp_left, des_left, cut_left_rgb, kp_right, des_right, cut_right_rgb, 0.5)    

    matches = matches * 2
    inliers, H = ransac(matches, 0.5, 2000)
    #inliers, H = ransac_warp(matches, 0.5, 2000)

    final_pic = stitch_img(left_rgb, right_rgb, H)
    final_pic = (final_pic * 255).astype('uint8') 
    final_bgr = cv2.cvtColor(final_pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output, final_bgr)

def find_homo(img1, img2):
    left_gray, left_origin, left_rgb = read_image(img1)
    right_gray, right_origin, right_rgb = read_image(img2)

    kp_left, des_left = SIFT(left_gray)
    kp_right, des_right = SIFT(right_gray)

    matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)####################### threshold=0.5
    # print(f"matches number = {len(matches)}")
    inliers, H = ransac(matches, 0.5, 5000)#################################################################### threshold=0.5
    return inliers, H
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [8, 8]
from stitch_functions import *
from PIL import Image

# panorama_size = (5, 5)
# refer_unit = (3, 3)
# input_directory = 'A12.6.1'
# output_directory = 'A12.6.1_cropped'

panorama_size = (4, 5)
refer_unit = (2, 3)
input_directory = 'kedr1'
output_directory = 'kedr1_undist'

top = np.array([[80, 176], [1063, 184], [2020, 181], [2987, 166]], dtype=np.float32)
bottom = np.array([[82, 2168], [1063, 2154], [2017, 2147], [2980, 2153]],dtype=np.float32)
imgpoints = np.concatenate((top, bottom), axis=0) # 2d points in image plane.

top_mean = np.mean(top[:, 1])
bottom_mean = np.mean(bottom[:, 1])
x_mean = np.mean(imgpoints[:, 0])
a = np.diff(top[:, 0])
b = np.diff(bottom[:, 0])
distance_mean = np.mean(np.concatenate((a, b)))

real_top = np.array([[-1.5*distance_mean + x_mean, top_mean, 0],
                    [-0.5*distance_mean + x_mean, top_mean, 0],
                    [ 0.5*distance_mean + x_mean, top_mean, 0],
                    [ 1.5*distance_mean + x_mean, top_mean, 0],], dtype=np.float32)
real_bottom = np.array([[-1.5*distance_mean + x_mean, bottom_mean, 0],
                    [-0.5*distance_mean + x_mean, bottom_mean, 0],
                    [ 0.5*distance_mean + x_mean, bottom_mean, 0],
                    [ 1.5*distance_mean + x_mean, bottom_mean, 0],], dtype=np.float32)
objpoints = np.concatenate((real_top, real_bottom), axis=0) # 3d point in real world space

# first_line = np.array([[137, 711],[383, 715],[631, 717],[874, 709],[1113, 712],[1351, 715],[1588, 716],
#               [1826, 704],[2066, 708],[2307, 710],[2546, 707],[2787, 697],[3029, 699],[3276, 698]], dtype=np.float32)
# second_line = np.array([[138, 1680],[385, 1683],[631, 1680],[874, 1671],[1113, 1668],[1350, 1676],[1587, 1670],
#               [1825, 1664],[2065, 1662],[2304, 1672],[2545, 1668],[2785, 1665],[3027, 1664],[3274, 1669]], dtype=np.float32)

# first_mean = np.mean(first_line[:, 1])
# second_mean = np.mean(second_line[:, 1])
# imgpoints2 = np.concatenate((first_line, second_line), axis=0)
# x_mean = np.mean(imgpoints2[:, 0])
# a = np.diff(first_line[:, 0])
# b = np.diff(second_line[:, 0])
# distance_mean = np.mean(np.concatenate((a, b)))

# imgpoints = np.concatenate((imgpoints, imgpoints2))

# real_first = np.array([[-6.5*distance_mean + x_mean, first_mean, 0],
#                     [-5.5*distance_mean + x_mean, first_mean, 0],
#                     [-4.5*distance_mean + x_mean, first_mean, 0],
#                     [-3.5*distance_mean + x_mean, first_mean, 0],
#                     [-2.5*distance_mean + x_mean, first_mean, 0],
#                     [-1.5*distance_mean + x_mean, first_mean, 0],
#                     [-0.5*distance_mean + x_mean, first_mean, 0],
#                     [ 0.5*distance_mean + x_mean, first_mean, 0],
#                     [ 1.5*distance_mean + x_mean, first_mean, 0],
#                     [ 2.5*distance_mean + x_mean, first_mean, 0],
#                     [ 3.5*distance_mean + x_mean, first_mean, 0],
#                     [ 4.5*distance_mean + x_mean, first_mean, 0],
#                     [ 5.5*distance_mean + x_mean, first_mean, 0],
#                     [ 6.5*distance_mean + x_mean, first_mean, 0],], dtype=np.float32)

# real_second = np.array([[-6.5*distance_mean + x_mean, second_mean, 0],
#                     [-5.5*distance_mean + x_mean, second_mean, 0],
#                     [-4.5*distance_mean + x_mean, second_mean, 0],
#                     [-3.5*distance_mean + x_mean, second_mean, 0],
#                     [-2.5*distance_mean + x_mean, second_mean, 0],
#                     [-1.5*distance_mean + x_mean, second_mean, 0],
#                     [-0.5*distance_mean + x_mean, second_mean, 0],
#                     [ 0.5*distance_mean + x_mean, second_mean, 0],
#                     [ 1.5*distance_mean + x_mean, second_mean, 0],
#                     [ 2.5*distance_mean + x_mean, second_mean, 0],
#                     [ 3.5*distance_mean + x_mean, second_mean, 0],
#                     [ 4.5*distance_mean + x_mean, second_mean, 0],
#                     [ 5.5*distance_mean + x_mean, second_mean, 0],
#                     [ 6.5*distance_mean + x_mean, second_mean, 0],], dtype=np.float32)

# objpoints = np.concatenate((objpoints, real_first, real_second))

objpoints = np.array([objpoints])
imgpoints = np.array([imgpoints])

img = cv2.imread('ruler.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

dist[0,2] = 0
dist[0,3] = 0

for i in range(panorama_size[0]):
    for j in range(panorama_size[1]):
        img = cv2.imread(f'{input_directory}/{i+1}.{j+1}.jpg')
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(f'{output_directory}/{i+1}.{j+1}_undistorced.jpg', dst)
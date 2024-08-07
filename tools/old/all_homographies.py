import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = [8, 8]
from stitch_functions import *

start_time = time.time()

output = 'row1_H.jpg'
H145 = find_homo('1.4.jpg', '1.5.jpg')
H134 = find_homo('1.3.jpg', '1.4.jpg')
H123 = find_homo('1.2.jpg', '1.3.jpg')
H112 = find_homo('1.1.jpg', '1.2.jpg')

H135 = H145 @  H134 
H125 = H135 @  H123 
H115 = H125 @  H112 

_, _, pic11 = read_image('1.1.jpg')
_, _, pic12 = read_image('1.2.jpg')
_, _, pic13 = read_image('1.3.jpg')
_, _, pic14 = read_image('1.4.jpg')
_, _, pic15 = read_image('1.5.jpg')

pics = np.stack((pic11, pic12, pic13, pic14, pic15), axis=0)
print("shape=", pics.shape)
n = 5
all_corners = np.empty((n, 4, 3))
for i in range(n):
    all_corners[i] = [[0, 0, 1], [pics[i].shape[1], 0, 1], [pics[i].shape[1], pics[i].shape[0], 1], [0, pics[i].shape[0], 1]]

all_homograpies = np.empty((n, 3, 3))
all_homograpies[0] = H115
all_homograpies[1] = H125
all_homograpies[2] = H135
all_homograpies[3] = H145
all_homograpies[4] = np.eye(3)

all_new_corners = np.empty((n, 4, 3))
for i in range(n):
    all_new_corners[i] = [np.dot(all_homograpies[i], corner) for corner in all_corners[i]]

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

ans = np.zeros((height_new, width_new, 3))
for i in range(n):
    warped_pic = cv2.warpPerspective(src=pics[i], M=T @ all_homograpies[i], dsize=size)
    ans = vector_stitching(ans, warped_pic)

final_pic = (ans).astype('uint8') 
final_bgr = cv2.cvtColor(final_pic, cv2.COLOR_RGB2BGR)
cv2.imwrite(output, final_bgr)
end_time = time.time()
print(f"1 row: {end_time - start_time}")


# start_time = time.time()
# elementary_stitching('2.5.jpg', 'row1.jpg', 'temp.jpg')
# elementary_stitching('2.4.jpg', 'temp.jpg', 'temp.jpg')
# elementary_stitching('2.3.jpg', 'temp.jpg', 'temp.jpg')
# elementary_stitching('2.2.jpg', 'temp.jpg', 'temp.jpg')
# elementary_stitching('2.1.jpg', 'temp.jpg', 'row2.jpg')
# end_time = time.time()
# print(f"2 row: {end_time - start_time}")
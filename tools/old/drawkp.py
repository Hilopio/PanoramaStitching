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
    img = cv2.drawKeypoints(rgb, kp, 0, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

left_gray, left_origin, left_rgb = read_image('1.1.jpg')
right_gray, right_origin, right_rgb = read_image('1.2.jpg')

kp_left, des_left = SIFT(left_gray)
kp_right, des_right = SIFT(right_gray)

kp_pic_left = plot_sift(left_gray, left_rgb, kp_left)
kp_pic_right = plot_sift(right_gray, right_rgb, kp_right)

kp_pic_left = cv2.cvtColor(kp_pic_left, cv2.COLOR_RGB2BGR)
kp_pic_right = cv2.cvtColor(kp_pic_right, cv2.COLOR_RGB2BGR)
cv2.imwrite('kp_1.jpg', kp_pic_left)
cv2.imwrite('kp_2.jpg', kp_pic_right)
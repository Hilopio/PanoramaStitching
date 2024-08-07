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
output_directory = 'kedr1_cropped'

n = 200
for i in tqdm(range(panorama_size[0])):
        for j in range (panorama_size[1]):
            pic = cv2.imread(f'{input_directory}/{i+1}.{j+1}.jpg')
            cropped_pic = pic[n:-n, n:-n]
            cv2.imwrite(f'{output_directory}/{i+1}.{j+1}.jpg', cropped_pic)
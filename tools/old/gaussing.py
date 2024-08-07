import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import shutil
import argparse

input_dir = "kedr1"
output_dir = "kedr1_filtered"


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('sigma')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sigma = float(args.sigma)

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    else:
        os.mkdir(output_dir)

    for filename in os.listdir(input_dir):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        image = cv2.imread(in_path)
        filtred_image = gaussian_filter(image, sigma)
        cv2.imwrite(out_path, filtred_image)


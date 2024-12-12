import cv2
import pickle
import numpy as np
import os, shutil
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
from multiprocessing import Pool

with open('camera_matrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)
with open('dist_coeffs.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)
with open('new_camera_matrix.pkl', 'rb') as f:
    new_camera_matrix = pickle.load(f)
with open('illumination_map.pkl', 'rb') as f:
    illumination_map = pickle.load(f)

def execute_on_all(in_global_dir, out_global_dir):
    if os.path.exists(out_global_dir):
        for dir in os.listdir(out_global_dir):
            file_path = os.path.join(out_global_dir, dir)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(out_global_dir)
    

    datasets = os.listdir(in_global_dir)
    in_dirs = [os.path.join(in_global_dir, ds) for ds in datasets]
    out_dirs = [os.path.join(out_global_dir, ds) for ds in datasets]
    args = list(zip(in_dirs, out_dirs))
    with Pool(3) as p:
        p.starmap(fix_dir, args)


def fix_dir(in_dir, out_dir):
    if os.path.exists(out_dir):
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(out_dir)

    for file in os.listdir(in_dir):
            in_path = os.path.join(in_dir, file)
            out_path = os.path.join(out_dir, file)
            # fix_image(in_path, out_path)
    print("complete!")
    

def objective_function(params, image, illumination_map):
    a, b = params
    corrected_image = (image / (a * illumination_map + b))
    ssim_value = -ssim(image, corrected_image, data_range=image.max() - image.min(), win_size=7, channel_axis=-1)
    regularization = (a - 1) ** 2 
    return ssim_value + regularization

def fix_image(in_path, out_path):
    image = cv2.imread(in_path)
    mask_3ch = np.repeat(illumination_map[..., None], 3, axis=2)
    initial_params = [1, 0.5]
    bounds = [(0.1, 2), (0, 1)]
    result = minimize(objective_function, initial_params, args=(image, mask_3ch), bounds=bounds)
    a_opt, b_opt = result.x
    corrected_image = image / (a_opt * mask_3ch + b_opt)
    corrected_image = cv2.normalize(corrected_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    undistorted_image = cv2.undistort(corrected_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    cv2.imwrite(out_path, undistorted_image)

if __name__ == '__main__':
    in_global_dir = 'LumenStone\data'
    out_global_dir = in_global_dir + '-calibrated'
    execute_on_all(in_global_dir, out_global_dir)
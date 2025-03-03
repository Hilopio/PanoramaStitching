import cv2
import pickle
import os
import shutil
import argparse
from tqdm import tqdm


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

    for file in tqdm(os.listdir(in_dir)):
        in_path = os.path.join(in_dir, file)
        out_path = os.path.join(out_dir, file)
        fix_image(in_path, out_path)


def fix_image(in_path, out_path):
    image = cv2.imread(in_path)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite(out_path, undistorted_image, [cv2.IMWRITE_JPEG_QUALITY, 98])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('calibration_files_dir', type=str)
    args = parser.parse_args()

    cm_path = os.path.join(args.calibration_files_dir, 'camera_matrix.pkl')
    with open(cm_path, 'rb') as f:
        camera_matrix = pickle.load(f)

    dc_path = os.path.join(args.calibration_files_dir, 'dist_coeffs.pkl')
    with open(dc_path, 'rb') as f:
        dist_coeffs = pickle.load(f)

    ncm_path = os.path.join(args.calibration_files_dir, 'new_camera_matrix.pkl')
    with open(ncm_path, 'rb') as f:
        new_camera_matrix = pickle.load(f)

    fix_dir(args.in_dir, args.out_dir)

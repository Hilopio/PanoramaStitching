import cv2
import pickle
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm


def fix_dir(in_dir, out_dir):
    """
    Processes all files in the input directory and saves the results in the output directory.
    If the output directory exists, its contents are cleared.

    Args:
        in_dir (Path): Path to the input directory containing images.
        out_dir (Path): Path to the output directory for saving processed images.
    """
    # If the output directory exists, clear its contents
    if out_dir.exists():
        for file_path in out_dir.iterdir():
            try:
                if file_path.is_file() or file_path.is_symlink():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # If the output directory does not exist, create it
        out_dir.mkdir(parents=True, exist_ok=True)

    # Process each file in the input directory
    for file in tqdm(in_dir.iterdir()):
        if file.is_file():
            out_path = out_dir / file.name
            fix_image(file, out_path)


def fix_image(in_path, out_path):
    """
    Corrects distortions in an image and saves the result.

    Args:
        in_path (Path): Path to the input image.
        out_path (Path): Path to save the corrected image.
    """
    print(in_path)  # Print the path to the current image
    image = cv2.imread(str(in_path))  # Load the image (convert Path to string for OpenCV)
    # Correct distortions using calibration data
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite(str(out_path), undistorted_image)  # Save the corrected image (convert Path to string for OpenCV)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correct distortions in images.")
    parser.add_argument('in_dir', type=str, help="Directory containing input images.")
    parser.add_argument('out_dir', type=str, help="Directory to save corrected images.")
    parser.add_argument('calibration_files_dir', type=str, help="Directory containing calibration files.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    calibration_files_dir = Path(args.calibration_files_dir)

    # Load calibration data
    cm_path = calibration_files_dir / 'camera_matrix.pkl'  # Path to the camera matrix file
    with open(cm_path, 'rb') as f:
        camera_matrix = pickle.load(f)

    dc_path = calibration_files_dir / 'dist_coeffs.pkl'  # Path to the distortion coefficients file
    with open(dc_path, 'rb') as f:
        dist_coeffs = pickle.load(f)

    ncm_path = calibration_files_dir / 'new_camera_matrix.pkl'  # Path to the new camera matrix file
    with open(ncm_path, 'rb') as f:
        new_camera_matrix = pickle.load(f)

    # Process images
    fix_dir(in_dir, out_dir)

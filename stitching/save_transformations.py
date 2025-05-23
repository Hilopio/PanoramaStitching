import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle


from image_alignment import Stitcher


def _load_images(img_paths):
    images = []
    for path in img_paths:
        img = Image.open(path)
        img = np.array(img).astype(np.float32) / 255
        images.append(img)

    return images


def stitch_collage(images, transforms, panorama_size):
    panorama = np.zeros((*panorama_size[::-1], 3), dtype=np.float32)
    for image, H in zip(images, transforms):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    return (panorama * 255).astype('uint8')


def find_and_save_transforms(input_dir, output_file, stchr):
    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]
    img_paths.sort()
    
    transforms, panorama_size, new_img_paths = stchr.only_transforms(img_paths=img_paths)

    data_to_save = {
        "transforms": transforms,
        "panorama_size": panorama_size,
        "img_paths": new_img_paths
    }

    with open(output_file, "wb") as f:
        pickle.dump(data_to_save, f)


if __name__ == '__main__':
    device = 'cuda:6'
    stchr = Stitcher(device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_global_dir', type=str)
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    input_global_dir = Path(args.input_global_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_global_dir.exists() and input_global_dir.is_dir():
        # Iterate over inner directories
        for inner_dir in tqdm(input_global_dir.iterdir()):
            if inner_dir.is_dir():
                try:
                    output_file = output_dir / Path(inner_dir.name + '-data.pkl')
                    find_and_save_transforms(inner_dir, output_file, stchr)
                except:
                    print(f"Error in directory {inner_dir}")
    else:
        print(f"Directory '{input_global_dir}' does not exist or is not a directory.")

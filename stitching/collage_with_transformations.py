import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle


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


def stitch_pano(transforms_file, output_file):

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)

    transforms = loaded_data["transforms"]
    panorama_size = loaded_data["panorama_size"]
    img_paths = loaded_data["img_paths"]

    pics = _load_images(img_paths)
    panorama_ans = stitch_collage(pics, transforms, panorama_size)

    output_img = Image.fromarray(panorama_ans)
    output_img.save(output_file, quality=95)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('transforms_dir', type=str)
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    transforms_dir = Path(args.transforms_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if transforms_dir.exists() and transforms_dir.is_dir():
        for transforms_file in tqdm(transforms_dir.iterdir()):
            output_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))
            stitch_pano(transforms_file, output_file)
    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")

import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# from image_alignment import Stitcher
from im_al import Stitcher


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


def stitch_pano(input_dir, output_file, stchr):
    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    transforms, panorama_size = stchr.only_transforms(img_paths=img_paths)
    pics = _load_images(stchr.img_paths)
    panorama_ans = stitch_collage(pics, transforms, panorama_size)  # stchr.transforms

    output_img = Image.fromarray(panorama_ans)
    output_img.save(output_file, quality=95)


if __name__ == '__main__':
    device = 'cuda:5'
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
                output_file = output_dir / Path(inner_dir.name + '-pano.jpg')
                stitch_pano(inner_dir, output_file, stchr)
    else:
        print(f"Directory '{input_global_dir}' does not exist or is not a directory.")

import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from image_alignment import Stitcher, find_warp_params


def stitch_collage(images, transforms, panorama_size):
    panorama = np.zeros((*panorama_size[::-1], 3))
    for image, H in zip(images, transforms):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    print(panorama[:50, :50])
    return panorama.astype('uint8')


def stitch_pano(input_dir, output_file, stchr):
    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    transforms = stchr.only_transforms(img_paths=img_paths)
    # T, panorama_size = find_translation_and_panorama_size(sizes, transformations)
    T, panorama_size, pics = find_warp_params(stchr.transforms, stchr.img_paths)
    transforms = [T @ H for H in transforms]

    panorama_ans = stitch_collage(pics, stchr.transforms, panorama_size)  # stchr.transforms
    print(panorama_ans.shape, panorama_ans.dtype)
    panorama_rgb = cv2.cvtColor(panorama_ans, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(panorama_rgb)
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

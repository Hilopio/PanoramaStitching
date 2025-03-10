import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle

borderValue = 1.0


def warp_img(img, H, panorama_size):
    warped_img = cv2.warpPerspective(
        img,
        H,
        panorama_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=borderValue
    )
    return warped_img


def warp_mask(mask, H, panorama_size):
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        panorama_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype('bool')
    return warped_mask


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


def find_color_scale(targetImg, queryImg, overlap_mask, gamma):
    eps = 1e-3
    targetColor = ((targetImg[overlap_mask] + eps) ** gamma).mean(axis=0)
    queryColor = ((queryImg[overlap_mask] + eps) ** gamma).mean(axis=0)
    print(f'targetColor = {targetColor}')
    return targetColor / queryColor


def stitch_lightcompensated_collage(images, transforms, panorama_size):
    gamma = 2.2
    eps = 1e-5
    color_scales = [(1, 1, 1),]
    panorama = warp_img(images[0], transforms[0], panorama_size)
    panorama_mask = warp_mask(np.ones(images[0].shape[:-1], dtype=int), transforms[0], panorama_size)
    for image, H in zip(images[1:], transforms[1:]):
        warped_mask = warp_mask(np.ones(image.shape[:-1], dtype=int), H, panorama_size)
        overlap_mask = panorama_mask & warped_mask
        assert np.any(overlap_mask)

        warped_img = warp_img(image, H, panorama_size)
        curr_color_scale = find_color_scale(panorama, warped_img, overlap_mask, gamma)
        color_scales.append(curr_color_scale)
        panorama_mask = np.where(panorama_mask, panorama_mask, warped_mask)
        panorama = np.where(
            warped_mask[..., np.newaxis],
            warped_img * (curr_color_scale + eps) ** (1 / gamma),
            panorama
        )

    print(color_scales)
    return (panorama.clip(0, 1) * 255).astype('uint8')


def stitch_pano(transforms_file, output_file):

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)

    transforms = loaded_data["transforms"]
    panorama_size = loaded_data["panorama_size"]
    img_paths = loaded_data["img_paths"]

    pics = _load_images(img_paths)
    panorama_ans = stitch_lightcompensated_collage(pics, transforms, panorama_size)

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

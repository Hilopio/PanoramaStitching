from pathlib import Path
from tqdm import tqdm
import shutil
import argparse

from utils import _load_images, _load_transforms, _save, _warp_collage
from gain_compensation_functions import gain_compensation, find_mean_color, compensate_mean_color


def stitch_collage_gaincomp(transforms_file, output_file):

    transforms, panorama_size, img_paths = _load_transforms(transforms_file)
    images = _load_images(img_paths)

    target_mean_color = find_mean_color(images)
    images = gain_compensation(images, transforms, panorama_size)
    new_mean_color = find_mean_color(images)
    color_scale = target_mean_color / (new_mean_color + 1e-6)
    images = [compensate_mean_color(img, color_scale) for img in images]

    pano = _warp_collage(images, transforms, panorama_size)
    _save(pano, output_file)


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

            collage_gaincomp_file = output_dir / \
                Path(transforms_file.name.replace("-data.pkl", "-gaincomp-collage.jpg"))
            stitch_collage_gaincomp(transforms_file, collage_gaincomp_file)

    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")

import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import _load_images, _load_transforms, _save, _warp_collage


def stitch_collage(transforms_file, output_file):

    transforms, panorama_size, img_paths = _load_transforms(transforms_file)
    pics = _load_images(img_paths)
    pano = _warp_collage(pics, transforms, panorama_size)
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
            output_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))
            stitch_collage(transforms_file, output_file)
    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")

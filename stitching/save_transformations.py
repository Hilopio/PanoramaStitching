import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle

from image_alignment import Stitcher


def find_and_save_transforms(input_dir, output_file, stchr):
    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    transforms, panorama_size, new_img_paths = stchr.only_transforms(img_paths=img_paths)

    data_to_save = {
        "transforms": transforms,
        "panorama_size": panorama_size,
        "img_paths": new_img_paths
    }

    with open(output_file, "wb") as f:
        pickle.dump(data_to_save, f)


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
                output_file = output_dir / Path(inner_dir.name + '-data.pkl')
                find_and_save_transforms(inner_dir, output_file, stchr)
    else:
        print(f"Directory '{input_global_dir}' does not exist or is not a directory.")

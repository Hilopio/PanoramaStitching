import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# from image_alignment import Stitcher
from im_al import Stitcher


def stitch_pano(input_dir, output_file, stchr):
    stchr.Stitch(input_dir, output_file)


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
        for inner_dir in tqdm(list(input_global_dir.iterdir())[1:]):
            if inner_dir.is_dir():
                output_file = output_dir / Path(inner_dir.name + '-pano.jpg')
                stitch_pano(inner_dir, output_file, stchr)
    else:
        print(f"Directory '{input_global_dir}' does not exist or is not a directory.")

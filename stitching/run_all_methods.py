import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

from stitch_collage import stitch_collage
from stitch_collage_gaincomp import stitch_collage_gaincomp
from stitch_graphcut import stitch_graphcut
from stitch_graphcut_gaincomp import stitch_graphcut_gaincomp
from stitch_full_pipeline import stitch_full_pipeline

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
            # try:
            # collage_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-collage.jpg"))
            # stitch_collage(transforms_file, collage_file)

            # collage_gaincomp_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-collage-gaincomp.jpg"))
            # stitch_collage_gaincomp(transforms_file, collage_gaincomp_file)

            # graphcut_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-graphcut.jpg"))
            # stitch_graphcut(transforms_file, graphcut_file)

            # graphcut_gaincomp_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-graphcut-gaincomp.jpg"))
            # stitch_graphcut_gaincomp(transforms_file, graphcut_gaincomp_file)

            full_pipeline_file = output_dir / Path(transforms_file.name.replace("-data.pkl", "-pano.jpg"))
            stitch_full_pipeline(transforms_file, full_pipeline_file)

            # except:
            #     print(f"Error while processing file {transforms_file}")

    else:
        print(f"Directory '{transforms_dir}' does not exist or is not a directory.")
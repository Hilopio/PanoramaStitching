import os, shutil
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from petroscope.panoramas.Stitching import Stitcher
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def execute_on_all(in_global_dir, out_dir, stchr):
    if os.path.exists(out_dir):
        for dir in os.listdir(out_dir):
            file_path = os.path.join(out_dir, dir)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.mkdir(out_dir)
    
    logger = {
        "series name"       : [],
        "images num"        : [],
        "preproc time"      : [],
        "LoFTR time"        : [],
        "homography time"   : [],
        "optimization time" : [],
        "stitching time"    : [],
        "total time"        : [],
        "num inliers"       : [],
        "initial error"     : [],
        "optimized error"   : []   
    }
    datasets = os.listdir(in_global_dir)
    for d in tqdm(datasets):
        in_dir = os.path.join(in_global_dir, d)
        out_file = os.path.join(out_dir, d + '.jpg')
        logger["series name"].append(d)
        stchr.Stitch(in_dir, out_file, logger=logger, verbose=False)
    
    df = pd.DataFrame(logger)
    df.to_csv("info.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_global_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stchr = Stitcher(device=DEVICE)

    execute_on_all(args.input_global_dir, args.output_dir, stchr)
import os
import argparse
import torch
from petroscope.panoramas.Stitching import Stitcher
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stchr = Stitcher(device=DEVICE)
    stchr.Stitch(args.input_dir, args.output_file)
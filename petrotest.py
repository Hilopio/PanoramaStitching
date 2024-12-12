import os
from petroscope.panoramas.Stitching import Stitcher

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

stchr = Stitcher()
in_dir = os.path.join('LumenStone', 'data-calibrated', 'ClPnA884')
out_file = 'test-pano.jpg'
stchr.Stitch(directory=in_dir, outputfile=out_file, logger=False, verbose=False)
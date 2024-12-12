from petroscope.panoramas.Stitching import Stitcher

stchr = Stitcher()
in_dir = 'LumenStone/ClPnA884'
out_file = 'test-pano.jpg'
stchr.Stitch(directory=in_dir, outputfile=out_file)
from PIL import Image
import pickle

from utils import _load_images
from graph_cutting import _warp_coarse_to_fine
from stitch_gain_compensated import gain_compensation, find_mean_color, compensate_mean_color

def stitch_gaincomp_graphcut(transforms_file, output_file):

    with open(transforms_file, "rb") as f:
        loaded_data = pickle.load(f)
        transforms = loaded_data["transforms"]
        panorama_size = loaded_data["panorama_size"]
        img_paths = loaded_data["img_paths"]

    images = _load_images(img_paths)
    target_mean_color = find_mean_color(images)
    images = gain_compensation(images, transforms, panorama_size)
    new_mean_color = find_mean_color(images)
    color_scale = target_mean_color / (new_mean_color + 1e-6)
    images = [compensate_mean_color(img, color_scale) for img in images]

    pano = _warp_coarse_to_fine(images, transforms, panorama_size)
    Image.fromarray(pano).save(output_file, quality=95)
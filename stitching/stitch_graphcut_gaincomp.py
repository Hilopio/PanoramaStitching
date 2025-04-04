from PIL import Image
import pickle

from utils import _load_images
from stitch_graphcut import _warp_coarse_to_fine
from stitch_collage_gaincomp import gain_compensation, find_mean_color, compensate_mean_color

def stitch_graphcut_gaincomp(transforms_file, output_file):

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

    pano, _ = _warp_coarse_to_fine(images, transforms, panorama_size)

    pano = (pano.clip(0, 1) * 255).astype('uint8')
    output_img = Image.fromarray(pano)
    output_img.save(output_file, quality=95)
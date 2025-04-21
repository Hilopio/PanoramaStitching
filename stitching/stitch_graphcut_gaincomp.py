from utils import _load_images, _load_transforms, _save
from stitch_graphcut import _warp_coarse_to_fine
from stitch_collage_gaincomp import gain_compensation, find_mean_color, compensate_mean_color


def stitch_graphcut_gaincomp(transforms_file, output_file):

    transforms, panorama_size, img_paths = _load_transforms(transforms_file)
    images = _load_images(img_paths)

    target_mean_color = find_mean_color(images)
    images = gain_compensation(images, transforms, panorama_size)
    new_mean_color = find_mean_color(images)
    color_scale = target_mean_color / (new_mean_color + 1e-6)
    images = [compensate_mean_color(img, color_scale) for img in images]

    pano, _ = _warp_coarse_to_fine(images, transforms, panorama_size)

    _save(pano, output_file)

from PIL import Image
from image_alignment import Stitcher
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


def _load_images(img_paths):
    images = []
    for path in img_paths:
        img = Image.open(path)
        img = np.array(img).astype(np.float32) / 255
        images.append(img)

    return images


def stitch_collage(images, transforms, panorama_size):
    panorama = np.zeros((*panorama_size[::-1], 3), dtype=np.float32)
    for image, H in tqdm(zip(images, transforms)):
        cv2.warpPerspective(
            image,
            H,
            panorama_size,
            panorama,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    return (panorama * 255).astype('uint8')


if __name__ == '__main__':
    input_dir = Path('../data/P1-calibrated/016')
    output_file = Path('../data/016-pano.jpg')

    img_paths = [
        img_p
        for img_p in input_dir.iterdir()
        if img_p.suffix in (".jpg", ".png")
    ]

    # img_paths.sort()

    device = 'cuda:5'
    stchr = Stitcher(device=device)

    transforms, panorama_size = stchr.only_transforms(img_paths=img_paths)
    pics = _load_images(stchr.img_paths)
    print(panorama_size)
    panorama_ans = stitch_collage(pics, transforms, panorama_size)  # stchr.transforms

    output_img = Image.fromarray(panorama_ans)
    output_img.save(output_file, quality=95)
    print('Completed')

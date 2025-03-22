import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

global_dir = Path('/home/data_repository/LumenStone/P1')
output_dir = Path('../data/avg_image.jpg')

sum_img = np.zeros((2547, 3396, 3), dtype=np.float32)
counter = 0

if global_dir.exists() and global_dir.is_dir():
    for inner_dir in tqdm(global_dir.iterdir()):
        if inner_dir.is_dir():
            for img_path in inner_dir.iterdir():
                if img_path.suffix in ['.png', '.jpg']:
                    sum_img += np.array(Image.open(img_path)) / 255
                    counter += 1

    sum_img /= counter
    sum_img *= 255
    sum_img = sum_img.astype(np.uint8)
    sum_img = Image.fromarray(sum_img)

    sum_img.save(output_dir, quality=95)
else:
    print(f"Directory '{global_dir}' does not exist or is not a directory.")
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path


def FFC(image, gain, black_field, output_path):

    corrected = (image - black_field) * gain

    corrected = (corrected.clip(0, 1) * 255).astype('uint8')
    Image.fromarray(corrected).save(output_path, quality=95)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory_path', type=str)
    parser.add_argument('white_field_path', type=str)
    parser.add_argument('black_field_path', type=str)
    parser.add_argument('corrected_directory_path', type=str)
    args = parser.parse_args()

    directory_path = Path(args.directory_path)
    white_field_path = Path(args.white_field_path)
    black_field_path = Path(args.black_field_path)
    corrected_directory_path = Path(args.corrected_directory_path)

    if white_field_path.name.endswith('.npy'):
        white_field = np.load(white_field_path)
        white_field = cv2.GaussianBlur(white_field, (501, 501), 0)
    elif white_field_path.name.endswith('.jpg') or white_field_path.name.endswith('.png'):
        white_field = Image.open(white_field_path)
        white_field = np.array(white_field, dtype='float32') / 255

        white_field = cv2.medianBlur(white_field, 5)
        white_field = cv2.GaussianBlur(white_field, (0, 0), 5)
    else:
        raise NotImplementedError
    

    black_field = Image.open(black_field_path)
    black_field = np.array(black_field, dtype='float32') / 255
    black_field = cv2.medianBlur(black_field, 5)
    black_field = cv2.GaussianBlur(black_field, (0, 0), 5)

    avg_white = white_field.mean(axis=(0, 1))
    avg_black = black_field.mean(axis=(0, 1))
    gain = (avg_white - avg_black) / (white_field - black_field)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    counter = 0
    for file_path in directory_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:

            image = Image.open(file_path)
            image = np.array(image) / 255

            output_path = corrected_directory_path / file_path.name

            relative_path = file_path.relative_to(directory_path)
            output_path = corrected_directory_path / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            FFC(image, gain, black_field, output_path)

            del image

            counter += 1
            if counter % 10 == 0:
                print(f'Processed {counter} images')

    print(f'Done! processed {counter} images')
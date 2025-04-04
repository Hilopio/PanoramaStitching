import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

def crop_image(image_path, output_path, crop_percent):
    """Обрезает изображение на crop_percent% с каждой стороны и сохраняет в output_path"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            crop_pixels = int(min(width, height) * crop_percent / 100)
            
            # Координаты для обрезки (left, upper, right, lower)
            box = (
                crop_pixels,
                crop_pixels,
                width - crop_pixels,
                height - crop_pixels
            )
            
            cropped_img = img.crop(box)
            cropped_img.save(output_path, qualiy=95)
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")

def process_directory(input_dir, output_dir, crop_percent):
    """Обрабатывает все изображения в директории и поддиректориях"""
    # Создаем список всех изображений с сохранением структуры
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / file)
    
    # Обрабатываем каждое изображение
    for image_path in tqdm(image_paths, desc="Обработка изображений"):
        # Создаем относительный путь для выходного файла
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        # Создаем папки если нужно
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Обрезаем и сохраняем изображение
        crop_image(image_path, output_path, crop_percent)

def main():
    parser = argparse.ArgumentParser(description='Обрезает изображения на n% с каждой стороны')
    parser.add_argument('input_dir', type=str, help='Входная директория с изображениями')
    parser.add_argument('output_dir', type=str, help='Выходная директория')
    parser.add_argument('crop_percent', type=float, help='Процент обрезки с каждой стороны (0-50)')
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if not 0 <= args.crop_percent < 50:
        print("Ошибка: crop_percent должен быть между 0 и 50")
        return
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Ошибка: Входная директория {input_dir} не существует")
        return
    
    # Очищаем или создаем выходную директорию
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Обрабатываем изображения
    process_directory(input_dir, output_dir, args.crop_percent)
    
    print(f"Обработка завершена. Результаты сохранены в {output_dir}")

if __name__ == "__main__":
    main()
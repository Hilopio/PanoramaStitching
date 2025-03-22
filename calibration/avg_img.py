import os
import cv2
import numpy as np

# Путь к директории с поддиректориями
base_dir = '../data/LumenStoneFull'

# Переменные для инкрементального усреднения
average_image = None
total_images = 0

# Рекурсивный обход всех поддиректорий
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Проверяем, что файл является изображением (по расширению)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Полный путь к файлу
            file_path = os.path.join(root, file)
            
            # Загрузка изображения
            img = cv2.imread(file_path)
            
            # Если изображение успешно загружено
            if img is not None:
                # Приведение изображения к float32 для точного усреднения
                img = img.astype(np.float32)
                
                # Если это первое изображение, инициализируем average_image
                if average_image is None:
                    average_image = img
                else:
                    # Инкрементальное обновление среднего
                    try:
                        average_image = (average_image * total_images + img) / (total_images + 1)
                        total_images += 1
                        if total_images % 100 == 0:
                            print(f"Обработано изображений: {total_images}")
                    except:
                        print(file_path)

                # Освобождаем память (опционально)
                del img

# Проверка, что найдены изображения
if average_image is None:
    print("Изображения не найдены.")
    exit()

# Преобразование результата обратно в uint8
average_image = np.clip(average_image, 0, 255).astype(np.uint8)

# Сохранение усредненного изображения
output_path = os.path.join(base_dir, 'average_image.jpg')
cv2.imwrite(output_path, average_image)

print(f"Усредненное изображение сохранено в: {output_path}")
print(f"Обработано изображений: {total_images}")
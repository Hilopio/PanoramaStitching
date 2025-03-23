import os
import cv2
import numpy as np

# Путь к директории с поддиректориями
base_dir = '../data/LumenStoneFull'

# Переменные для инкрементального усреднения
sum_image = None
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
                img = img / 255
                
                # Если это первое изображение, инициализируем average_image
                if sum_image is None:
                    sum_image = img
                else:
                    # Инкрементальное обновление среднего
                    try:
                        sum_image = sum_image + img 
                        total_images += 1
                        if total_images % 100 == 0:
                            print(f"Обработано изображений: {total_images}")
                    except:
                        print(file_path)

                # Освобождаем память (опционально)
                del img

# Проверка, что найдены изображения
if sum_image is None:
    print("Изображения не найдены.")
    exit()

output_path = os.path.join(base_dir, 'average_image.npy')
avg_array = cv2.cvtColor(sum_image.astype(np.float32), cv2.COLOR_BGR2RGB)
avg_array = np.clip(sum_image / total_images, 0, 1)
np.save(output_path, avg_array)

print(f"Усредненное изображение сохранено в: {output_path}")
print(f"Обработано изображений: {total_images}")
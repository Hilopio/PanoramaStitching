 # Калибровка датасета
 хорошим, но опциональным шагом перед созданием панорамы, будет исправление искажений съемки.
 для этого можно воспользоваться скриптами из директории calibration

 ## неравномерное освещение
 неравномерное освещение камерой микроскопа создает эффект виньетирования, что плохо скажется 
 на итоговой панораме. Для исправления этого эффекта можно использовать метод flat-field correction,
 необходимо получить два изображения:
 white_field - фотография чисто белого фона в тех же условиях, для получения карты освещенности
 black_field - фотография с закрытым объективом, для компенсации особенностей восприятия матрицы камеры

 после чего в каждом изображении датасета (предполагается что они все сфотографированны одной камерой в одних условиях)
 будет скомпенсирована освещенность по формуле

 gain = (average_color(white_field) - average_color(black_field)) / (white_field - black_field)
 corrected_image = (image - black_field) * gain

 для выполнения этого шага можно воспользоваться скриптом compensate_luminance.py

 python3 compensate_luminance.py /path/to/dataset /path/to/white_field /path/to/black_field.jpg /path/to/compensated_dataset

 где 
 white_field - .npy массив форма которого совпадает с формой изображения, в котором хранится карта освещенности в диапазоне [0, 1]
 black_field - изображение в форматах .jpg, .png, ...
 Пример:
 python3 compensate_luminance.py /home/data_repository/LumenStone/P1 calibration_params/average_image.npy calibration_params/L1.1.jpg ../data/P1-lumcomp

 ## оптическая дисторсия
даже самая хорошая камера, из-за своей системы линз вносит геометрические искажения. основной эфеект которых можно свести к радиальной дисторсии.

 для выполнения этапа можно воспользоваться .sh скриптом run_undistort_all.sh
 
 ./run_undistort_all.sh ../data/P1-lumcomp ../data/P1-calibrated-ffc calibration_params


 # создание панорам
 ## у меня есть набор изображений, я не хочу ни в чем разбираться, я просто хочу панораму
 ## создание коллажей
 ## глобальная компенсация освещенности
 ## удаление швов
 ## блендинг
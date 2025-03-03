#!/bin/bash

# Usage: ./run_lightcorrect_all.sh INPUT_DIR OUTPUT_DIR MIRROR_FILE
#
# Description:
#   This script corrects illumination distortion for all files in subdirectories of INPUT_DIR.
#   It uses MIRROR_FILE as a reference for illumination distortion.
#   Results are saved in OUTPUT_DIR.
#
# Arguments:
#   INPUT_DIR    - Input directory containing subdirectories to process.
#   OUTPUT_DIR   - Output directory to save the corrected results.
#   MIRROR_FILE  - Path to the mirror file used for illumination correction.
#
# Example:
#   ./run_lightcorrect_all.sh /home/data_repository/LumenStone/P1 ~/pano/data/P1-lightfixed ./calibration_params/mirror1200.jpg
#
# May need:
# sed -i 's/\r$//' ./run_lightcorrect_all.sh

# Проверка количества аргументов
if [ "$#" -ne 3 ]; then
  echo "Использование: $0 <INPUT_DIR> <OUTPUT_DIR> <MIRROR_FILE>"
  exit 1
fi

# Аргументы командной строки
INPUT_DIR="$1"
OUTPUT_DIR="$2"
MIRROR_FILE="$3"

# Создать выходную директорию, если она не существует
mkdir -p "$OUTPUT_DIR"

# Перебор всех поддиректорий в INPUT_DIR
for DIR in "$INPUT_DIR"/*; do
  # Проверка, что это директория
  if [ -d "$DIR" ]; then
    # Извлечение имени директории (basename)
    DIR_NAME=$(basename "$DIR")
    
    # Создание поддиректории в OUTPUT_DIR
    OUTPUT_SUBDIR="$OUTPUT_DIR/$DIR_NAME"
    mkdir -p "$OUTPUT_SUBDIR"
    
    # Запуск Python-скрипта
    echo "Обработка директории: $DIR"
    python -m petroscope.calibrate.run -i "$DIR" -o "$OUTPUT_SUBDIR" -m "$MIRROR_FILE"
    
    echo "Завершена обработка: $DIR"
    echo "Результаты сохранены в: $OUTPUT_SUBDIR"
    echo "--------------------------"
  fi
done

echo "Все директории обработаны."
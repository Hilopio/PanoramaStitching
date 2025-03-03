#!/bin/bash

# Usage: ./run_undistort_all.sh INPUT_DIR OUTPUT_DIR DATA_DIR
#
# Description:
#   This script corrects radial distortion for all files in subdirectories of INPUT_DIR.
#   It uses distortion information from DATA_DIR.
#   Results are saved in OUTPUT_DIR.
#
# Arguments:
#   INPUT_DIR    - Input directory containing subdirectories to process.
#   OUTPUT_DIR   - Output directory to save the undistorted results.
#   DATA_DIR     - Directory containing distortion information (e.g., calibration data).
#
# Example:
#   ./run_undistort_all.sh /path/to/input /path/to/output /path/to/data
#
# May need:
# sed -i 's/\r$//' ./run_undistort_all.sh


# Проверка количества аргументов
if [ "$#" -ne 3 ]; then
  echo "Использование: $0 <INPUT_DIR> <OUTPUT_DIR> <DATA_DIR>"
  exit 1
fi

# Аргументы командной строки
INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATA_DIR="$3"

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
    python undistort.py "$DIR" "$OUTPUT_SUBDIR" "$DATA_DIR"
    
    echo "Завершена обработка: $DIR"
    echo "Результаты сохранены в: $OUTPUT_SUBDIR"
    echo "--------------------------"
  fi
done

echo "Все директории обработаны."
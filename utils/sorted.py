import os
import shutil

# Исходная папка
source_dir = r"../data/ubuntu_flat1"

# Целевые папки
upper_dir = r"../data/raw/upper"
lower_dir = r"../data/raw/lower"

# Создаём целевые папки, если их нет
os.makedirs(upper_dir, exist_ok=True)
os.makedirs(lower_dir, exist_ok=True)

# Проходим по всем файлам в исходной папке
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)

    # Проверяем, что это файл
    if not os.path.isfile(source_path):
        continue

    # Копируем файлы по условиям
    if "-maxillary" in filename:
        shutil.copy2(source_path, os.path.join(upper_dir, filename))

    elif "-mandibular" in filename:
        shutil.copy2(source_path, os.path.join(lower_dir, filename))

print("Готово!")

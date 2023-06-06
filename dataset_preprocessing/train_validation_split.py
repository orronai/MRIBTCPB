import os
import random
import shutil

def move_random_files(source_folder, destination_folder, percentage):
    file_list = os.listdir(source_folder)
    num_files = len(file_list)
    num_files_to_move = int(num_files * percentage)

    random_files = random.sample(file_list, num_files_to_move)

    for file_name in random_files:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.move(source_file, destination_file)
        print(f"Moved file: {file_name}")


classes = ['notumor', 'glioma', 'meningioma', 'pituitary']
source_folder = '../input/training/'
destination_folder = '../input/validation/'
percentage = 0.2  # 20%

for each in classes:
    source_folder_each = source_folder + each
    destination_folder_each = destination_folder + each
    move_random_files(source_folder_each, destination_folder_each, percentage)

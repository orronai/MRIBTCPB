import os
import numpy as np
from PIL import Image

# Define the path to the folder containing grayscale images
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
folder_path = '../input/training/'

image_files = []
for each in classes:
    image_class_path = folder_path + each
    temp = [os.path.join(image_class_path, file) for file in os.listdir(image_class_path)]
    image_files += temp


# Initialize variables to accumulate pixel values
sum_pixels = np.zeros((1,))
sum_squared_pixels = np.zeros((1,))
total_pixels = 0

# Iterate over each image file
for file_path in image_files:
    # Open the image file and convert it to grayscale
    image = Image.open(file_path).convert('L')
    # Convert the image to a NumPy array
    pixels = np.array(image) / 255
    # Flatten the image array
    flattened_pixels = pixels.flatten()

    # Accumulate pixel values
    sum_pixels += np.sum(flattened_pixels)
    sum_squared_pixels += np.sum(np.square(flattened_pixels))
    total_pixels += flattened_pixels.shape[0]

# Calculate the mean and standard deviation
mean = sum_pixels / total_pixels
variance = (sum_squared_pixels / total_pixels) - np.square(mean)
std_dev = np.sqrt(variance)

print("Mean:", mean)
print("Standard Deviation:", std_dev)

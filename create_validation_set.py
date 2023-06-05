import os
from sklearn.model_selection import train_test_split
import shutil

# Set the paths to the train, test, and validation folders
data_dir = 'Lungs_Dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
validation_dir = os.path.join(data_dir, 'val')

# Create the validation directory if it doesn't exist
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
#os.makedirs(validation_dir, exist_ok=True)

# Get the subfolder names (classes)
subfolders = [subfolder for subfolder in os.listdir(train_dir) 
              if os.path.isdir(os.path.join(train_dir, subfolder))]
# print(subfolders) # ['COVID19', 'PNEUMONIA', 'NORMAL']

# Initialize lists to store the image paths and corresponding labels
images = []
labels = []

# Collect the image paths and labels from the training set
for subfolder in subfolders:
    subfolder_dir = os.path.join(train_dir, subfolder)
    image_files = [file for file in os.listdir(subfolder_dir) if file.endswith('.jpg')]
    images.extend([os.path.join(subfolder_dir, file) for file in image_files])
    labels.extend([subfolder] * len(image_files))

# Perform stratified train-test split to create the validation set
train_images, val_images, train_labels, val_labels = train_test_split(images, 
                                                                      labels, 
                                                                      test_size=0.2, 
                                                                      stratify=labels)
# print(len(train_images)) # 4115
# print(len(val_images)) # 1029

# Move the validation images to the validation directory
for image_path, label in zip(val_images, val_labels):
    label_dir = os.path.join(validation_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    shutil.move(image_path, label_dir)

import os
import sys
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


from keras.models import Sequential, Model
from keras.layers import (
    Dense, Conv2D, MaxPool2D, Dropout, Flatten, 
    BatchNormalization, GlobalAveragePooling2D
)

from keras.applications.densenet import DenseNet121
from keras import backend as K
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
"""
# packages versions
print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('TensorFlow: {}'.format(tf.__version__))
# To check if we have access to the GPU on my Apple M1 Mac
print("TensorFlow has access to the following devices:")
for device in tf.config.list_physical_devices():
    print(f"· {device}")
"""

print(f"{os.listdir('./')}")

# number of images in each folder
print(f"train NORMAL: {len(os.listdir('./Lungs_Dataset/train/NORMAL'))}")
print(f"test NORMAL: {len(os.listdir('./Lungs_Dataset/test/NORMAL'))}")

print(f"train PNEUMONIA: {len(os.listdir('./Lungs_Dataset/train/PNEUMONIA'))}")
print(f"test PNEUMONIA: {len(os.listdir('./Lungs_Dataset/test/PNEUMONIA'))}")

print(f"train COVID19: {len(os.listdir('./Lungs_Dataset/train/COVID19'))}")
print(f"test COVID19: {len(os.listdir('./Lungs_Dataset/test/COVID19'))}")









import pydicom
import matplotlib.pyplot as plt
import os

def load_data(url: str):
    pass

def find_images(url: str):
    # Iterate over each file in the directory
    for filename in os.listdir(url):
        file_path = os.path.join(url, filename)
        
        if filename.endswith('.dcm'):  # Only process DICOM files
            
            # Load the DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Access the pixel data
            pixel_data = ds.pixel_array
            
            # Display the image
            plt.imshow(pixel_data, cmap=plt.cm.gray)
            plt.axis('off')  # Optional: remove axes
            plt.title(filename)  # Optional: display filename as the title
            plt.show()
        elif os.path.isdir(file_path):
            find_images(file_path)


# Path to the directory containing DICOM files
directory = 'Body_Parts_Dataset/train'

find_images(directory)

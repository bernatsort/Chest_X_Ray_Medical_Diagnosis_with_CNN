import pydicom
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(url: str):
    return pd.read_csv(url)

def find_images(url: str):
    
    images = []
    
    # Iterate over each file in the directory
    for filename in os.listdir(url):
        file_path = os.path.join(url, filename)
        
        if filename.endswith('.dcm'):  # Only process DICOM files
            
            # Load the DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Access the pixel data
            pixel_data = ds.pixel_array
            
            image = cv2.imread(filename)
            
            # Display the image
            #plt.imshow(pixel_data, cmap=plt.cm.gray)
            #plt.axis('off')  # Optional: remove axes
            #plt.title(filename)  # Optional: display filename as the title
            #plt.show()
            return image
        elif os.path.isdir(file_path):
            new_data = find_images(file_path)
            if new_data is not None:
                images.append(new_data)

def complete_code(main_dir: str, csv_dir: str):
    
    tags_df = load_data(csv_dir)

    # Load the images and collect image names, tags, and labels
    image_list = []
    image_names = []
    image_labels = []
    
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                dcm = pydicom.dcmread(dcm_path)
                image_array = dcm.pixel_array

                image_list.append(image_array.flatten())
                image_names.append(file)
                
    for i in range(len(image_names)):
        for row in tags_df.index:
            if tags_df.loc[row, "SOPInstanceUID"] in image_names[i]:
                image_labels.append(tags_df.loc[row, "Target"])

    # Convert the image list and labels to NumPy arrays
    X = np.array(image_list)
    y = np.array(image_labels)

    # Perform label encoding for the target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print("Confusion Matrix:")
    #print(cm)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    # Set labels, title, and ticks
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    # Show the plot
    plt.show()

def complete_code_2(main_dir: str, csv_dir: str):
    tags_df = load_data(csv_dir)

    # Load the images and collect image names, tags, and labels
    image_list = []
    image_names = []
    image_labels = []
    
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                dcm = pydicom.dcmread(dcm_path)
                image_array = dcm.pixel_array

                image_list.append(image_array.flatten())
                image_names.append(file)
                
    for i in range(len(image_names)):
        for row in tags_df.index:
            if tags_df.loc[row, "SOPInstanceUID"] in image_names[i]:
                image_labels.append(tags_df.loc[row, "Target"])
                
    # Convert the image list and labels to NumPy arrays
    X = np.array(image_list)
    y = np.array(image_labels)

    # Standardize the feature vectors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering using K-means
    k = 21
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)

    # Get the cluster labels
    cluster_labels = kmeans.labels_
    tags_df['Kmeans'] = cluster_labels
    
    print(tags_df)

    # Assign cluster labels to body parts
    cluster_assigned_labels = {}
    for i, label in enumerate(y):
        if label not in cluster_assigned_labels:
            cluster_assigned_labels[label] = cluster_labels[i]

    # Print the assigned cluster labels for each body part
    for label, cluster_label in cluster_assigned_labels.items():
        print("Body Part:", label)
        print("Cluster Label:", cluster_label)
        print()

def classification_VGG(main_dir: str, csv_dir: str):
    
    tags_df = load_data(csv_dir)
    
    target_size=(224,224,3)

    # Load the images and collect image names, tags, and labels
    image_list = []
    image_names = []
    image_labels = []
    
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                dcm = pydicom.dcmread(dcm_path)
                image_array = dcm.pixel_array
                
                resized_image = cv2.resize(image_array, target_size[:2])
                if len(target_size) == 3 and resized_image.ndim == 2:
                    #resized_image = np.expand_dims(resized_image, axis=-1)
                    resized_image = np.expand_dims(resized_image, axis=-1)
                    resized_image = np.concatenate([resized_image] * 3, axis=-1)  # Convert grayscale to RGB
                
                image_list.append(resized_image)
                #image_list.append(image_array.flatten())
                image_names.append(file)
                
                #image = np.array(image_array).shape
                #print(image)
                
                #cv2.imshow("Title", image_array.flatten())
                #cv2.waitKey(0)
                
    for i in range(len(image_names)):
        for row in tags_df.index:
            if tags_df.loc[row, "SOPInstanceUID"] in image_names[i]:
                image_labels.append(tags_df.loc[row, "Target"].split(' ')[0])
                tags_df.loc[row, "Target"] = tags_df.loc[row, "Target"].split(' ')[0]

    print(tags_df)
    print(tags_df['Target'].unique())
    
    # Convert the image list and labels to NumPy arrays
    X = np.array(image_list)
    print(X.shape)
    #X = np.array([tf.image.resize(np.expand_dims(image, axis=0), target_size).numpy() for image in image_list])

    #X = np.array([tf.image.resize(image, target_size).numpy() for image in image_list])
    y = np.array(image_labels)

    # Perform label encoding for the target labels
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    y_encoded = label_encoder.fit_transform(y)
    y_encoded = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.8, random_state=0)
    
    tr_datagen = ImageDataGenerator()
    te_datagen = ImageDataGenerator()
    
    # Set the batch size
    batch_size = 32

    # Resize train data
    #X_train = np.array([tf.image.resize(image, target_size).numpy() for image in X_train])
    #X_test = np.array([tf.image.resize(image, target_size).numpy() for image in X_test])

    # Create data generators
    traindata = tr_datagen.flow(X_train, y_train, batch_size=batch_size)
    testdata = te_datagen.flow(X_test, y_test, batch_size=batch_size)

    # Layers
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    #model.add(Dense(units=2, activation="softmax"))
    model.add(Dense(units=22, activation="softmax"))

    # Optimizer
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    
    # Validations checkpoints and Early stopping
    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    print(X_train.shape)


    model.summary()
    #hist = model.fit(traindata, steps_per_epoch=25, validation_data=testdata, validation_steps=10, epochs=5, callbacks=[checkpoint, early])
    hist = model.fit(traindata, steps_per_epoch=5, validation_data=testdata, validation_steps=2, epochs=5)
    
    
    # Plot train history
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show()
    

train_csv_file = "Body_Parts_Dataset/train.csv"
#train_data = load_data(train_csv_file)

directory = 'Body_Parts_Dataset/train'
#images_list = find_images(directory)

classification_VGG(directory, train_csv_file)
complete_code(directory, train_csv_file)
#complete_code_2(directory, train_csv_file)
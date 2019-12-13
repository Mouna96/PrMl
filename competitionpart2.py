

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import cv2
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

path = "C:/Work/sgndataset/train/"
class_names = sorted(os.listdir(path)) # need to correct

base_model = tf.keras.applications.mobilenet.MobileNet(
    input_shape = (224,224,3),
    include_top = False)

base_model.summary() # listing of the network structure

in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0] 
  # Grab the input of base model out_tensor = base_model.outputs[0]
  # Grab the output of base model
  # Add an average pooling layer (averaging each of the 1024 channels):
  
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)
# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs = [in_tensor], outputs = [out_tensor])
# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model. model.compile(loss = "categorical_crossentropy", optimizer = ’sgd’)

# Find all image files in the data directory.

X = []  # Feature vectors will go here.
y = []  # Class ids will go here.

for root, dirs, files in os.walk(r"C:/Work/sgndataset/train/"):
    for name in files:
            # Load the image:
        if name.endswith(".jpg"):
            img = plt.imread(root + os.sep + name)
            
            # Resize it to the net input size:
            img = cv2.resize(img, (224, 224))
            
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            
            # Push the data through the model:
#            x = model.predict(img[np.newaxis, ...])[0]
            
            # And append the feature vector to our list.
            X.append(img)
            
            # Extract class name from the directory name:
            label = root.split('/')[-1]
            y.append(class_names.index(label))

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)
num_classes = 17
#np.save('X_data',X)
#np.save('y_data',y)
X=np.load('Ximg_data.npy')
Y=np.load('yimg_data.npy')


X_train, X_tst, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
####

# CALCULATE THE WEIGHTS 

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
class_weights = dict(enumerate(class_weights))

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)
# fits the model on batches with real-time data augmentation:




# Classifiers

#base_model = tensorflow.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3),include_top = False, alpha=0.25)
#base_model = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3),
#                                                                    alpha=1.0, include_top=False,
#                                                                    weights='imagenet', input_tensor=None, pooling=None, classes=17)
base_model = tensorflow.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                    input_tensor=None, input_shape=(224,224,3), pooling=None, classes=17)




in_tensor = base_model.inputs[0] # Grab the input of base model
# Grab the output of base model
out_tensor = base_model.outputs[0]
out_tensor =tensorflow.keras.layers.Flatten()(out_tensor)
out_tensor =tensorflow.keras.layers.Dense(100, activation='relu')(out_tensor)
out_tensor =tensorflow.keras.layers.Dense(17,activation='softmax')(out_tensor)
model = tensorflow.keras.models.Model(inputs = [in_tensor],outputs = [out_tensor])
model.summary()
batch_size = 50

epochs = 30

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=len(X_train) /128, epochs=15)

model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,
           class_weight=class_weights,verbose=1,validation_data=(X_tst, y_test))
score = model.evaluate(X_tst, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])


from pathlib import Path



model=load_model('incep_trial_weights.h5')


basepath = Path("C:/Work/sgndataset/testset/")
files_in_basepath = basepath.iterdir()
with open("submission.csv", "w") as fp:
    fp.write("Id,Category\n")
    for image  in files_in_basepath:
#        print(image)
        img = plt.imread(image)
        image=str(image).split('\\')[-1]
        image=image.split('.')[0]
        img = cv2.resize(img, (224, 224))
#
        img = img.astype(np.float32)
        img -= 128
#
        pred=model.predict(img[np.newaxis, ...])[0]
        index = np.argmax(pred, axis=0)
            # Extract class name from the directory name:
        label = class_names[index]
        fp.write("%d,%s\n" % (int(image), label))










# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
import os
import sys 

import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
            x = model.predict(img[np.newaxis, ...])[0]
            
            # And append the feature vector to our list.
            X.append(x)
            
            # Extract class name from the directory name:
            label = root.split('/')[-1]
            y.append(class_names.index(label))

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

np.save('X_data',X)
np.save('y_data',y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# make classifiers
LDA = LinearDiscriminantAnalysis()
SVM = SVC()
SVM_RBF = SVC(kernel='rbf',gamma='scale')
LogReg = LogisticRegression()
RFC = RandomForestClassifier(n_estimators=100,max_depth=100)

# fit classifiers
LDA.fit(X_train, y_train)
SVM.fit(X_train, y_train)
SVM_RBF.fit(X_train, y_train)
LogReg.fit(X_train, y_train)
RFC.fit(X_train, y_train)

# predict results
prediction_LDA = LDA.predict(X_test)
prediction_SVM = SVM.predict(X_test)
prediction_SVM_RBF = SVM_RBF.predict(X_test)
prediction_LogReg = LogReg.predict(X_test)
prediction_RFC = RFC.predict(X_test)

# accurac score of classifiers
acc_LDA = accuracy_score(y_test, prediction_LDA)
acc_SVM = accuracy_score(y_test, prediction_SVM)
acc_SVM_RBF = accuracy_score(y_test, prediction_SVM_RBF)
acc_LogReg = accuracy_score(y_test, prediction_LogReg)
acc_RFC = accuracy_score(y_test, prediction_RFC)

print("Accuracy of LDA is: ", acc_LDA)
print("Accuracy of SVM is: ", acc_SVM)
print("Accuracy of SVM_RBF is: ", acc_SVM_RBF)
print("Accuracy of LogReg is: ", acc_LogReg)
print("Accuracy of RFC is: ", acc_RFC)








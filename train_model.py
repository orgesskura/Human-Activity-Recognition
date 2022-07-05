#!/usr/bin/env python3
# coding: utf-8

# # Introduction to Human Activity Recognition

# Here we will explore some common ways of preprocessing human activity recognition data.
# 
# Using the example data we will learn:
# * how to merge multiple files into one large DataFrame
# * how to split data into sliding windows
# * how to quickly extract features from a window
# * how to set the number of classes considered for classification
# * how to build a simple Random Forest Classifier and train it on HAR data
# * how to build a simple CNN and train it on HAR data 
# 
# Bear in mind that the sample data offered is not cleaned or high quality. You should not use it in your own experiments but it is useful for this tutorial.
# 
# You will need the following packages: 
# * tsfresh
# * scikit-learn
# * tensorflow

# #### Basic imports


import pandas as pd
import numpy as np
import tsfresh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
import os
#import seaborn as sns
import matplotlib.pyplot as plt
# keras goodies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics as kmetrics
import tensorflow.keras.backend as K
from sklearn.metrics import precision_recall_fscore_support
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# ## Loading multiple files into one large DataFrame

# At this stage you should only be working with clean data, saved in the format required for Coursework 1. An example of such data can be found in the Data/Clean/ folder.
base_df = pd.DataFrame()
#notice there are 3 clean folders. Clean has 8 Respeck data. Clean2 has all Respeck data. Clean3 has all Thingy data
clean_data_folder = "./Data/Respeck"

for filename in os.listdir(clean_data_folder):
    full_path = f"{clean_data_folder}/{filename}"
    #print(full_path)
    if full_path == "./Data/.DS_Store" :
        continue
    # load data into a DataFrame
    new_df = pd.read_csv(full_path )
    # merge into the base DataFrame
    base_df = pd.concat([base_df, new_df])


base_df.reset_index(drop=True, inplace=True)



# Now you can get a list of all recording ids, activities, sensor types and anything else you might need.


print(f"The data was collected using the sensors: {base_df.sensor_type.unique()}")
print(f"The data was collected for the activities: {base_df.activity_type.unique()}")
print(f"The number of unique recordings is: {len(base_df.recording_id.unique())}")
print(f"The subject IDs in the recordings are: {len(base_df.subject_id.unique())}")
subject_ids = base_df.subject_id.unique()
subject_ids = np.array(sorted(subject_ids))
print(len(subject_ids))

# You can of course change the clean data folder to where you keep all the PDIoT data and you should be seeing a lot more subject IDs, unique recordings and activity types.


#column of interest for respeck
columns_of_interest = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']



window_size = 30# 50 datapoints for the window size, which, at 25Hz, means 2 seconds
step_size = 15# this is 50% overlap

window_number = 0 # start a counter at 0 to keep track of the window number

all_overlapping_windows = []

for rid, group in base_df.groupby("recording_id"):
    #print(f"Processing rid = {rid}")
    
    large_enough_windows = [window for window in group.rolling(window=window_size, min_periods=window_size) if len(window) == window_size]
    
    overlapping_windows = large_enough_windows[::step_size]
    removed_activities = ['Climbing stairs', 'Descending stairs' ,'Desk work','Movement','Climbing_stairs','Descending_stairs','Desk_work']
    flag = False
    for act in removed_activities:
        if rid.find(act) != -1:
            flag = True
            break
    if flag:
        continue
    print(f"Processing rid = {rid}")
    #print(overlapping_windows)
    # then we will append a window ID to each window
    for window in overlapping_windows:
        window.loc[:, 'window_id'] = window_number
        window_number += 1
    if len(overlapping_windows) == 0 :
        continue
    all_overlapping_windows.append(pd.concat(overlapping_windows).reset_index(drop=True))




final_sliding_windows = pd.concat(all_overlapping_windows).reset_index(drop=True)

class_labels = {
    'Sitting':0,
    'Sitting bent forward':0,
    'Sitting bent backward':0,
    'Standing':0,
    'Walking at normal speed' : 1,
    'Running':2,
    'Lying down on back':3,
    'Lying down on stomach':3,
    'Lying down left':3,
    'Lying down right':3,
    'Falling on the back' : 4,
    'Falling on knees':4,
    'Falling on the left':4,
    'Falling on the right':4
  
}

window_id_class_labels = final_sliding_windows.groupby("window_id")[['activity_type']].agg(np.min).replace(class_labels)


filters = 64
kernel_size = 3
n_features = 6
activation='relu'
n_classes = 5




X = [ [] for _ in range(30) ]
y = [ [] for _ in range(30) ]
for window_id, group in final_sliding_windows.groupby('window_id'):
    #print(f"window_id = {window_id}")
    subject_id = group['subject_id'].iloc[0]
    #print(subject_id)
    #print(subject_ids[0])
    index = np.where(subject_ids == subject_id)[0]
    index=index[0]
    shape = group[columns_of_interest].values.shape
    #print(f"shape = {shape}")
    
    X[index].append(group[columns_of_interest].values)
    y[index].append(class_labels[group["activity_type"].values[0]])
    
    
test_accuracies = []
test_precision = []
test_recall = []
test_f1_score = []
test_per_class = [[0 for j in range(4)] for i in range(5)]


for i in range(30):
    model = Sequential()
    
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='linear', 
                 input_shape=(window_size, n_features)))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='linear'))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='linear'))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    X_test,y_test = X[i] , y[i]
    X_test = np.asarray(X_test)
    if len(X_test) == 0:
        continue
    X_train, y_train = [],[]
    for j in range(30):
        if j != i:
            X_train.extend(X[j]) 
            y_train.extend(y[j])
        
    X_train = np.asarray(X_train)
    y_train = np.asarray(pd.get_dummies(y_train), dtype=np.float32)
    y_test = np.asarray(pd.get_dummies(y_test), dtype=np.float32) 
    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics = ['accuracy'])
    model.fit(X_train, y_train,
        batch_size=64, epochs=50)
    # stats
    y_pred_ohe = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_ohe, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    
    print("*" * 80)
    print("Classification report")
    print("*" * 80)
    print(classification_report(y_true_labels, y_pred_labels))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    test_accuracies.append(test_acc)
    precision,recall,f1_score,_ = precision_recall_fscore_support(y_true_labels, y_pred_labels,average='weighted')
    test_precision.append(precision)
    test_recall.append(recall)
    test_f1_score.append(f1_score)
    matrix = np.array(precision_recall_fscore_support(y_true_labels, y_pred_labels,average=None)).T
    test_per_class += matrix
    
    
for item in test_accuracies:
    print(item)

average_precision = sum(test_precision) / len(test_precision)
average_recall = sum(test_recall) / len(test_recall)
average_f_score = sum(test_f1_score) / len(test_f1_score)
average_score = sum(test_accuracies) / len(test_accuracies)
print('Average Testing Accuracy is {}, average precision is {}, average recall is {},average fscore is {}'.format(average_score, average_precision,average_recall,average_f_score ))
print(test_per_class/30)
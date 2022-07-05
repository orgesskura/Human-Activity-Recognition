#!/usr/bin/env python3
# coding: utf-8



import pandas as pd
import numpy as np
import tsfresh
from sklearn.metrics import classification_report
import tensorflow as tf
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
# keras goodies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics as kmetrics
import tensorflow.keras.backend as K

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

def evaluate_model(model_path, test_data_path):
    base_df = pd.read_csv(test_data_path,dtype=object)

    base_df.reset_index(drop=True, inplace=True)
    subject_ids = base_df.subject_id.unique()
    subject_ids = np.array(sorted(subject_ids))
    columns_of_interest = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    window_size = 30# 50 datapoints for the window size, which, at 25Hz, means 2 seconds
    step_size = 15# this is 50% overlap

    window_number = 0 # start a counter at 0 to keep track of the window number

    all_overlapping_windows = []

    for rid, group in base_df.groupby("recording_id"):
    
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
    'Falling on the left':4,
    'Falling on the right':4,
    'Falling on the back': 4,
    'Falling on knees':4
    }

    window_id_class_labels = final_sliding_windows.groupby("window_id")[['activity_type']].agg(np.min).replace(class_labels)

    

    X = []
    y = []

    for window_id, group in final_sliding_windows.groupby('window_id'):
        shape = group[columns_of_interest].values.shape
        X.append(group[columns_of_interest].values)
        y.append(class_labels[group["activity_type"].values[0]])

    X_test = np.asarray(X)
    y = np.asarray(y)
    y_test = np.asarray(pd.get_dummies(y), dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_data = np.array(X_test, dtype=np.float32)
    predictions = []
    #print()
    for i in range(input_data.shape[0]):
        #print(input_data[i:(i+1),:,:].shape)
        interpreter.set_tensor(input_details[0]['index'], input_data[i:(i+1),:,:])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.squeeze(output_data))
   
    predictions = np.array(predictions)
    y_pred_labels = np.argmax(predictions, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    print("*" * 80)
    print("Classification report")
    print("*" * 80)
    print(classification_report(y_true_labels,y_pred_labels))


     # #show confusion matrix
    proper_labels = {'Sitting/Standing': 0,
    'Walking': 1,
    'Running': 2,
    'Lying down': 3,
    'Falling': 4
    }
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    matrix = np.array(precision_recall_fscore_support(y_true_labels, y_pred_labels,average=None)).T
    print('Sitting/Standing .............: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'
      .format(cm_norm[0][0],matrix[0][0],matrix[0][1],matrix[0][2]))
    print('Walking .............: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'
      .format(cm_norm[1][1],matrix[1][0],matrix[1][1],matrix[1][2]))
    print('Running .............: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'
      .format(cm_norm[2][2],matrix[2][0],matrix[2][1],matrix[2][2]))
    print('Lying down .............: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'
      .format(cm_norm[3][3],matrix[3][0],matrix[3][1],matrix[3][2]))
    print('Falling .............: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'
      .format(cm_norm[4][4],matrix[4][0],matrix[4][1],matrix[4][2]))
    
    fig = plt.figure(figsize=(10, 8))
    cm2 = np.round((cm_norm * 1000)) / 1000
    sns.heatmap(cm2, xticklabels=proper_labels, yticklabels=proper_labels, 
            annot=True, fmt='g')
    plt.xlabel('Predicted Labels',fontsize=16)
    plt.ylabel('True Labels',fontsize=16)
    #plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    fig.savefig('confusion_matrix.png', facecolor=fig.get_facecolor(), edgecolor='none',dpi=300)
    plt.show()
    sys.exit(0)
evaluate_model(sys.argv[1], sys.argv[2])
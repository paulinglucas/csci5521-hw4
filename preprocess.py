import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import PIL

data_shape = [150,150]

train_files = np.asarray(list(glob.glob("chest_xray/train/NORMAL/*.jpeg")) + list(glob.glob("chest_xray/train/PNEUMONIA/*.jpeg")))
np.random.shuffle(train_files)

image = PIL.Image.open(train_files[0]).convert('L').resize(data_shape)
image.save("test.jpeg")

n_train = len(train_files)
X_train = np.zeros([n_train,data_shape[0]*data_shape[1]])
y_train = np.zeros(n_train, dtype = np.int32)
for i in range(n_train):
    x = PIL.Image.open(train_files[i]).convert('L')
    x = np.array(x.resize(data_shape))
    X_train[i] = x.reshape([x.shape[0]*x.shape[1],]).astype('float32') / 255.
    if "PNEUMONIA" in train_files[i]: y_train[i] = 1
    elif "NORMAL" in train_files[i]: y_train[i] = 0
    else: print("ERROR with file: " + str(train_files[i]))

data = np.c_[X_train, y_train]
np.random.shuffle(data)
X_train = data[:,:-1]
y_train = data[:,-1]

np.savetxt("X_train.csv", X_train, delimiter = ",")
np.savetxt("y_train.csv", y_train, delimiter = ",")

test_files = list(glob.glob("chest_xray/test/NORMAL/*.jpeg")) + list(glob.glob("chest_xray/test/PNEUMONIA/*.jpeg"))

n_test = len(test_files)
X_test = np.zeros([n_test,data_shape[0]*data_shape[1]])
y_test = np.zeros(n_test, dtype = np.int32)
for i in range(n_test):
    x = PIL.Image.open(test_files[i]).convert('L')
    x = np.array(x.resize(data_shape))
    X_test[i] = x.reshape([x.shape[0]*x.shape[1],]).astype('float32') / 255.
    if "PNEUMONIA" in test_files[i]: y_test[i] = 1
    elif "NORMAL" in test_files[i]: y_test[i] = 0
    else: print("ERROR with file: " + str(test_files[i]))

data = np.c_[X_test, y_test]
np.random.shuffle(data)
X_test = data[:,:-1]
y_test = data[:,-1]

np.savetxt("X_test.csv", X_test, delimiter = ",")
np.savetxt("y_test.csv", y_test, delimiter = ",")

val_files = list(glob.glob("chest_xray/val/NORMAL/*.jpeg")) + list(glob.glob("chest_xray/val/PNEUMONIA/*.jpeg"))

n_val = len(val_files)
X_val = np.zeros([n_val,data_shape[0]*data_shape[1]])
y_val = np.zeros(n_val, dtype = np.int32)
for i in range(n_val):
    x = PIL.Image.open(val_files[i]).convert('L')
    x = np.array(x.resize(data_shape))
    X_val[i] = x.reshape([x.shape[0]*x.shape[1],]).astype('float32') / 255.
    if "PNEUMONIA" in val_files[i]: y_val[i] = 1
    elif "NORMAL" in val_files[i]: y_val[i] = 0
    else: print("ERROR with file: " + str(val_files[i]))

data = np.c_[X_val, y_val]
np.random.shuffle(data)
X_val = data[:,:-1]
y_val = data[:,-1]

np.savetxt("X_val.csv", X_val, delimiter = ",")
np.savetxt("y_val.csv", y_val, delimiter = ",")

print("Sample Size: " + str(len(y_test)+len(y_train)+len(y_val)))
print("Train Size: " + str(len(y_train)))
print("Test Size: " + str(len(y_test)))
print("Validation Size: " + str(len(y_val)))

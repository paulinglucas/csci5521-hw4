# CSCI 5521 HW#4
#### a. Dataset name: Chest X-Ray Images (Pneumonia)

> Dataset Link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

> Sample Size: 5,840

> Trainging Size: 5,216

> Testing Size: 624

> Validation Size: 16
   
#### b. The features of this data set are pixel values. The target variable is a binary variable; 1 = Pneumonia, 0 = Normal/Healthy

#### c. Data cleaning and engineering:

> This dataset is already split into Training, Testing, and Validation sets -- we'll leave them split as is.

> The sample images will be opened as arrays using the Pillow module.

> The sample images don't have a consistent dimension, so we will resize the images to a standard size dxd (probably 150x150 pixels).

> Almost all sample images are grayscale, but some seem to be colored -- we'll convert all to grayscale.

> Then the sample arrays will be flatted to be one-dimensional arrays of length 150x150 = 22,500

> Each set (training, testing, and validation) is split into NORMAL and PNEUMONIA subdirectories. The samples in the NORMAL directories will be assigned the label 0 and sameples in the PNEUMONIA directories will be assigned the label 1.

#### 1. Run preprocessor.py to generation the X_train.csv, y_train.csv, X_test.csv, y_test.csv, X_val.csv, and y_val.csv
#### 2. Run myLinearSVM.py to run the linear svm on the .csv files created by the preprocessor (validation part in progress)
#### 3. Run myKernelSVM.py to run the kernel svm on the .csv files created by the preprocessor (validation part in progress)

#### NOTE:

> the input image dimension used is set in preprocessor.py; currently I'm using 150x150, but this does take a few minutes to run.

> the number of training images to use is set in myLinearSVM.py/myKernelSVM.py. I've typically been using a few hundred for the sake of time.

> validation is in progress, AKA I haven't done it yet. So far I just loop through the possible C values and create a model/output for each.


> I've been getting about 75% accuracy overall for both the Linear and Kernel models
1.
   a. Dataset name: Chest X-Ray Images (Pneumonia)
      Dataset Link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
      Sample Size: 5,840
      Trainging Size: 5,216
      Testing Size: 624
      Validation Size: 16
   b. The features of this data set are pixel values. The target variable is a binary variable; 1 = Pneumonia, 0 = Normal/Healthy
   c. Data cleaning and engineering:
	This dataset is already split into Training, Testing, and Validation sets -- we'll leave them split as is.
	The sample images will be opened as arrays using the Pillow module.
	The sample images don't have a consistent dimension, so we will resize the images to a standard size dxd (probably 150x150 pixels).
	Almost all sample images are grayscale, but some seem to be colored -- we'll convert all to grayscale.
	Then the sample arrays will be flatted to be one-dimensional arrays of length 150x150 = 22,500
	Each set (training, testing, and validation) is split into NORMAL and PNEUMONIA subdirectories. The samples in the NORMAL directories will be assigned the label 0 and sameples in the PNEUMONIA directories will be assigned the label 1.

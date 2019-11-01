import argparse
import numpy as np
np.set_printoptions(precision=5, suppress=True)

def open_train_file(training_image_path,training_label_path):
    file_train_image = open(training_image_path, 'rb')
    file_train_label = open(training_label_path, 'rb')

    global train_image_magic, train_image_number, train_image_row, train_image_col, train_label_magic, train_label_total_count
    train_image_magic = int.from_bytes(file_train_image.read(4), byteorder = 'big')         # magic number in image training file
    train_image_number = int.from_bytes(file_train_image.read(4), byteorder = 'big')        # number of images in training image file
    train_image_row = int.from_bytes(file_train_image.read(4), byteorder = 'big')           # number of rows in training image file
    train_image_col = int.from_bytes(file_train_image.read(4), byteorder = 'big')           # number of columns in training image file
    train_label_magic = int.from_bytes(file_train_label.read(4), byteorder = 'big')         # magic number in training label file
    train_label_total_count = int.from_bytes(file_train_label.read(4), byteorder = 'big')   # number of items in training label file

    return file_train_image, file_train_label

def open_file():
	data_type = np.dtype("int32").newbyteorder('>')
	
	data = np.fromfile("./train-images.idx3-ubyte", dtype = "ubyte")
	train_image = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28 * 28).transpose()
	train_image_bin = np.divide(train_image, 128).astype("int")

	train_label = np.fromfile("./train-labels.idx1-ubyte",dtype = "ubyte").astype("int")
	train_label = train_label[2 * data_type.itemsize : ].reshape(1, 60000)
	return train_image_bin, train_label

if __name__ == '__main__':

    train_bin, train_label = open_file()

    
import argparse
import numpy as np
np.set_printoptions(precision=5, suppress=True)

# py hw2_1.py --TRAINING_IMAGE=train-images.idx3-ubyte --TRAINING_LABEL=train-labels.idx1-ubyte --TESTING_IMAGE=t10k-images.idx3-ubyte --TESTING_LABEL=t10k-labels.idx1-ubyte --OPTION=0

LABEL_NUM = 10

def open_train_file(training_image_path, training_label_path):
    file_train_image = open(training_image_path, 'rb')
    file_train_label = open(training_label_path, 'rb')

    global train_image_magic, train_image_number, train_image_row, train_image_col, train_label_magic, train_label_total_count
    train_image_magic = int.from_bytes(file_train_image.read(4), byteorder='big')         # magic number in image training file
    train_image_number = int.from_bytes(file_train_image.read(4), byteorder='big')        # number of images in training image file
    train_image_row = int.from_bytes(file_train_image.read(4), byteorder='big')           # number of rows in training image file
    train_image_col = int.from_bytes(file_train_image.read(4), byteorder='big')           # number of columns in training image file
    train_label_magic = int.from_bytes(file_train_label.read(4), byteorder='big')         # magic number in training label file
    train_label_total_count = int.from_bytes(file_train_label.read(4), byteorder='big')   # number of items in training label file

    return file_train_image, file_train_label

def open_test_file(testing_image_path, testing_label_path):
    file_test_image = open(testing_image_path, 'rb')
    file_test_label = open(testing_label_path, 'rb')

    global test_image_magic, test_image_number, test_image_row, test_image_col, test_label_magic, test_label_total_count
    test_image_magic = int.from_bytes(file_test_image.read(4), byteorder='big')         # magic number in image testing file
    test_image_number = int.from_bytes(file_test_image.read(4), byteorder='big')        # number of images in testing image file
    test_image_row = int.from_bytes(file_test_image.read(4), byteorder='big')           # number of rows in testing image file
    test_image_col = int.from_bytes(file_test_image.read(4), byteorder='big')           # number of columns in testing image file
    test_label_magic = int.from_bytes(file_test_label.read(4), byteorder='big')         # magic number in testing label file
    test_label_total_count = int.from_bytes(file_test_label.read(4), byteorder='big')   # number of items in testing label file

    return file_test_image, file_test_label

def normalization(probability):
    temp = 0
    for j in range(LABEL_NUM):
        temp += probability[j]
    for j in range(LABEL_NUM):
        probability[j] /= temp
    return probability

def print_result(probability, answer):
    print("Posterior (in log scale):")
    for j in range(LABEL_NUM):
        print(j, ": ", probability[j])
    prediction = np.argmin(probability)
    print("Prediction: ", prediction, ", Ans: ", answer)
    print("")
    if prediction == answer:
        return 0
    else:
        return 1

def print_imagination_discrete(likelihood):
    print("Imagination of numbers in Bayesian Classifier:")
    print("")
    for i in range(LABEL_NUM):
        print(i, ":")
        for j in range(28):
            for k in range(28):
                temp = 0
                #bin 0 ~ 16 vs bin 16 ~ 32
                for t in range(16):
                    temp += likelihood[i][j * 28 + k][t]
                for t in range(16, 32):
                    temp -= likelihood[i][j * 28 + k][t]
                if temp > 0:
                    print("0", end = " ")
                else:
                    print("1", end = " ")
            print("")
        print("")

def discrete_mode(training_image_path, training_label_path, testing_image_path, testing_label_path):
    file_train_image, file_train_label = open_train_file(training_image_path, training_label_path)
    prior = np.zeros((LABEL_NUM), dtype=int)
    likelihood = np.zeros((LABEL_NUM, train_image_row*train_image_col, 32), dtype=int)

    for img_cnt in range(train_image_number):
        label = int.from_bytes(file_train_label.read(1), byteorder='big')
        prior[label] += 1
        for pixel_idx in range(train_image_row*train_image_col):
            pixel_value = int.from_bytes(file_train_image.read(1), byteorder='big')
            likelihood[label][pixel_idx][int(pixel_value/8)] += 1

    # Testing data
    file_test_image, file_test_label = open_test_file(testing_image_path, testing_label_path)

    likelihood_sum = np.zeros((LABEL_NUM, train_image_row*train_image_col), dtype=int)
    for i in range(LABEL_NUM):
        for j in range(train_image_row*train_image_col):
            for k in range(32):
                likelihood_sum[i][j] += likelihood[i][j][k]

    error = 0
    for i in range(test_image_number):
        # print("NOW IS : {}".format(i))
        answer = int.from_bytes(file_test_label.read(1), byteorder='big')
        probability = np.zeros((LABEL_NUM), dtype = float)
        test_image = np.zeros((test_image_row*test_image_col), dtype=int)
        for j in range(test_image_row*test_image_col):
            test_image[j] = int((int.from_bytes(file_test_image.read(1), byteorder='big'))/8)
        for j in range(LABEL_NUM):
            probability[j] += np.log(float(prior[j]/train_image_number))
            for k in range(test_image_row*test_image_col):
                temp = likelihood[j][k][test_image[k]]
                if temp == 0:   #psuedo count
                    probability[j] += np.log(float(1e-6 / likelihood_sum[j][k]))
                else:
                    probability[j] += np.log(float(likelihood[j][k][test_image[k]] / likelihood_sum[j][k]))

        probability = normalization(probability)
        error += print_result(probability, answer)
    print_imagination_discrete(likelihood)
    print("Error rate: ", float(error/test_image_number))

def Gaussian_distribution(value, mean, var):
    return np.log(1.0 / (np.sqrt(2.0 * np.pi * var))) - ((value - mean)**2.0 / (2.0 * var))

def print_imagination_continuous(likelihood):
    print("Imagination of numbers in Bayesian Classifier:")
    print("")
    for i in range(LABEL_NUM):
        print(i, ":")
        for j in range(28):
            for k in range(28):
                if likelihood[i][j * 28 + k] < 128:
                    print("0", end = " ")
                else:
                    print("1", end = " ")
            print("")
        print("")

def continuous_mode(training_image_path, training_label_path, testing_image_path, testing_label_path):
    file_train_image, file_train_label = open_train_file(training_image_path, training_label_path)
    prior = np.zeros((LABEL_NUM), dtype = float)
    pixel_square = np.zeros((LABEL_NUM, train_image_row*train_image_col), dtype=float)
    pixel_mean = np.zeros((LABEL_NUM, train_image_row*train_image_col), dtype=float)
    pixel_var = np.zeros((LABEL_NUM, train_image_row*train_image_col), dtype=float)

    for img_cnt in range(train_image_number):
        label = int.from_bytes(file_train_label.read(1), byteorder='big')
        prior[label] += 1
        for pixel_idx in range(train_image_row * train_image_col):
            pixel_value = int.from_bytes(file_train_image.read(1), byteorder='big')
            pixel_square[label][pixel_idx] += (pixel_value**2)
            pixel_mean[label][pixel_idx] += pixel_value

    #Calculate mean and standard deviation
    for label in range(LABEL_NUM):
        for pixel_idx in range(train_image_row*train_image_col):
            pixel_mean[label][pixel_idx] = float(pixel_mean[label][pixel_idx] / prior[label])
            pixel_var[label][pixel_idx] = float(pixel_square[label][pixel_idx] / prior[label]) - float(pixel_mean[label][pixel_idx] ** 2)
    
            # psuedo count for variance
            if pixel_var[label][pixel_idx] == 0:
                pixel_var[label][pixel_idx] = 1e-4
    prior = prior / train_image_number
    prior = np.log(prior)
    
    #Testing data
    file_test_image, file_test_label = open_test_file(testing_image_path, testing_label_path)

    error = 0
    for image_idx in range(10000):
        # print("NOW IS : {}".format(image_idx))
        answer = int.from_bytes(file_test_label.read(1), byteorder='big')
        probability = np.zeros((LABEL_NUM), dtype = float)
        test_image = np.zeros((28*28), dtype = float)
        for pixel_idx in range(28*28):
            test_image[pixel_idx] = int.from_bytes(file_test_image.read(1), byteorder='big')
        for label in range(LABEL_NUM):
            probability[label] += prior[label]
            for pixel_idx in range(test_image_row * test_image_col):
                testing_value = Gaussian_distribution(test_image[pixel_idx], pixel_mean[label][pixel_idx], pixel_var[label][pixel_idx])
                probability[label] += testing_value

        # print("probability = {}".format(probability))
        probability = normalization(probability)
        error += print_result(probability, answer)
    print_imagination_continuous(pixel_mean)
    print("Error rate: ", float(error / test_image_number))

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--TRAINING_IMAGE", type = str  )
    parser.add_argument("--TRAINING_LABEL", type = str  )
    parser.add_argument("--TESTING_IMAGE", type = str  )
    parser.add_argument("--TESTING_LABEL", type = str  )
    parser.add_argument("--OPTION", type = int  )
    args = parser.parse_args()
    training_image_path = args.TRAINING_IMAGE
    training_label_path = args.TRAINING_LABEL
    testing_image_path = args.TESTING_IMAGE
    testing_label_path = args.TESTING_LABEL
    toggle_mode = args.OPTION

    if toggle_mode == 0:
        discrete_mode(training_image_path, training_label_path, testing_image_path, testing_label_path)
    else:
        continuous_mode(training_image_path, training_label_path, testing_image_path, testing_label_path)
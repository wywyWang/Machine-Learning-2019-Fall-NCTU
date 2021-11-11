import argparse
import numpy as np
import numba as nb
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=5, suppress=True)

# py EM_algorithm.py

SEED = 42
np.random.seed(SEED)

# @nb.jit
def open_file():
	data_type = np.dtype("int32").newbyteorder('>')
	
	data = np.fromfile("./train-images.idx3-ubyte", dtype="ubyte")
	train_image = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28*28)
	train_image_bin = np.divide(train_image, 128).astype("int")

	train_label = np.fromfile("./train-labels.idx1-ubyte", dtype="ubyte").astype("int")
	train_label = train_label[2 * data_type.itemsize:].reshape(60000, 1)
	return train_image_bin, train_label

@nb.jit
def Estep(train_bin, mu, pi, Z):
    # E step
    for img_idx in range(img_cnt):
        num_sum = np.full((num_cnt, 1), 1, dtype=np.float64)
        for num_idx in range(num_cnt):
            for pixel_idx in range(pixel_cnt):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx][0] *= mu[num_idx][pixel_idx]
                else:
                    num_sum[num_idx][0] *= (1-mu[num_idx][pixel_idx])
            num_sum[num_idx][0] *= pi[num_idx][0]

        marginal_num = np.sum(num_sum)
        if marginal_num == 0:
            marginal_num = 1
        
        for num_idx in range(num_cnt):
            Z[img_idx][num_idx] = num_sum[num_idx][0] / marginal_num

    return Z

# @nb.jit
def Mstep(train_bin, mu, pi, Z):
    N_cluster = np.sum(Z, axis=0)
    for num_idx in range(num_cnt):
        for pixel_idx in range(pixel_cnt):
            sum_pixel = np.dot(train_bin[:, pixel_idx], Z[:, num_idx])
            marginal = N_cluster[num_idx]
            if marginal == 0:
                marginal = 1
            mu[num_idx][pixel_idx] = sum_pixel / marginal

        pi[num_idx][0] = N_cluster[num_idx] / img_cnt
        if pi[num_idx][0] == 0:
            pi[num_idx][0] = 1

    return mu, pi

@nb.jit
def difference(mu, mu_prev):
    diff = 0
    for num_idx in range(num_cnt):
        for pixel_idx in range(pixel_cnt):
            diff += abs(mu[num_idx][pixel_idx] - mu_prev[num_idx][pixel_idx])
    return diff

# @nb.jit
def print_imagination(mu):
    mu_new = mu.copy()
    for num_idx in range(num_cnt):
        print("\nclass: ", num_idx)
        for pixel_idx in range(pixel_cnt):
            if pixel_idx % 28 == 0 and pixel_idx != 0:
                print("")
            if mu_new[num_idx][pixel_idx] >= 0.5:
                print("1", end=" ")
            else:
                print("0", end=" ")
        print("")

@nb.jit
def label_cluster(train_bin, train_label, mu, pi):
    table = np.zeros(shape=(num_cnt, num_cnt), dtype=np.int)
    for img_idx in range(img_cnt):
        num_sum = np.full((num_cnt), 1, dtype=np.float64)
        for num_idx in range(num_cnt):
            for pixel_idx in range(pixel_cnt):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= mu[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - mu[num_idx][pixel_idx])
            num_sum[num_idx] *= pi[num_idx][0]
        table[train_label[img_idx][0]][np.argmax(num_sum)] += 1

    relation = np.full((num_cnt), -1, dtype=np.int)
    print(table)
    for num_idx in range(num_cnt):
        ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
        relation[ind[0]] = ind[1]           # label ind[0]th is ind[1] cluster
        for i in range(num_cnt):
            table[i][ind[1]] = -1
            table[ind[0]][i] = -1

    return relation

def print_label(relation, mu):
    for num_idx in range(num_cnt):
        cluster = relation[num_idx]
        print("\nLabeled class : ", num_idx)
        for pixel_idx in range(pixel_cnt):
            if pixel_idx % 28 == 0 and pixel_idx != 0:
                print("")
            if mu[cluster][pixel_idx] >= 0.5:
                print("1", end=" ")
            else:
                print("0", end=" ")
        print("")

@nb.jit
def print_confusion_matrix(train_bin, train_label, mu, pi, relation):
    error = img_cnt
    confusion_matrix = np.zeros(shape=(num_cnt, 4), dtype=np.int)       # TP FP TN FN
    for img_idx in range(img_cnt):
        num_sum = np.full((num_cnt), 1, dtype=np.float64)
        for num_idx in range(num_cnt):
            for pixel_idx in range(pixel_cnt):
                if train_bin[img_idx][pixel_idx] == 1:
                    num_sum[num_idx] *= mu[num_idx][pixel_idx]
                else:
                    num_sum[num_idx] *= (1 - mu[num_idx][pixel_idx])
            num_sum[num_idx] *= pi[num_idx][0]

        predict_cluster = np.argmax(num_sum)
        predict_label = np.where(relation==predict_cluster)

        for num_idx in range(num_cnt):
            if num_idx == train_label[img_idx][0]:
                if num_idx == predict_label[0]:
                    error -= 1
                    confusion_matrix[num_idx][0] += 1
                else:
                    confusion_matrix[num_idx][3] += 1
            else:
                if num_idx == predict_label[0]:
                    confusion_matrix[num_idx][1] += 1
                else:
                    confusion_matrix[num_idx][2] += 1

    for num_idx in range(num_cnt):
        print("Confusion matrix {}:".format(num_idx))
        print("\t\tPredict number {}\tPredict not number {}".format(num_idx, num_idx))
        print("Is number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][0], confusion_matrix[num_idx][3]))
        print("Isn't number {}\t\t{}\t\t\t{}".format(num_idx, confusion_matrix[num_idx][1], confusion_matrix[num_idx][2]))
        print("Sensitivity (Successfully predict number {}): {}".format(num_idx, confusion_matrix[num_idx][0] / (confusion_matrix[num_idx][0] + confusion_matrix[num_idx][3])))
        print("Specificity (Successfully predict not number {}): {}".format(num_idx, confusion_matrix[num_idx][2] / (confusion_matrix[num_idx][2] + confusion_matrix[num_idx][1])))
        print("---------------------------------------------------------------\n")

    return error

if __name__ == '__main__':
    train_bin, train_label = open_file()
    global num_cnt, pixel_cnt, img_cnt
    num_cnt = 10
    pixel_cnt = 28 * 28
    img_cnt = 60000
    epochs = 10

    print("train bin size = {}".format(train_bin.shape))

    pi = np.random.random_sample((num_cnt, 1))
    mu = np.random.random_sample((num_cnt, pixel_cnt))
    mu_prev = np.zeros((num_cnt, pixel_cnt), dtype=np.float64)
    Z  = np.random.random_sample((img_cnt, 10)) 

    for step in range(epochs):
        Z = Estep(train_bin, mu, pi, Z)
        mu, pi = Mstep(train_bin, mu, pi, Z)
        gap = difference(mu, mu_prev)

        print_imagination(mu)
        print("No. of Iteration: {}, Difference: {}\n".format(step+1, gap))
        print("---------------------------------------------------------------\n")

        mu_prev = mu.copy()

    relation = label_cluster(train_bin, train_label, mu, pi)
    print(relation)
    print_label(relation, mu)
    print("---------------------------------------------------------------\n")
    error = print_confusion_matrix(train_bin, train_label, mu, pi, relation)
    print("Total iteration to converge: {}".format(epochs))
    print("Total error rate: {}".format(error / img_cnt))
from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

num = 100
epochs = 15
K = 2
gamma_c = 1 / (255*255)
gamma_s = 1 / (100*100)

def read_input(filename):
    img = Image.open(filename)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width*height, 3))

    coord = np.array([]).reshape(0, 2)
    for i in range(num):
        row_x = np.full(num, i)
        row_y = np.arange(num)
        row = np.array(list(zip(row_x, row_y))).reshape(1*num, 2)
        coord = np.vstack([coord, row])

    return pixel, coord

def compute_kernel(color, coord):
    length = len(color)
    gram_matrix = np.zeros((length, length))
    spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
    spatial_rbf = np.exp(-gamma_s*spatial_sq_dists)
    color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
    color_rbf = np.exp(-gamma_c*color_sq_dists)
    kernel = spatial_rbf * color_rbf

    return kernel

def initial(data, initial_method):
    prev_classification = np.random.randint(K, size=data.shape[0])
    if initial_method == 'random':
        C_x = np.random.randint(0, num, size=K)
        C_y = np.random.randint(0, num, size=K)
        mu = np.array([])
        if K == 3:
            C_z = np.random.randint(0, num, size=K)
            mu = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
        elif K == 4:
            C_z = np.random.randint(0, num, size=K)
            C_w = np.random.randint(0, num, size=K)
            mu = np.array(list(zip(C_x, C_y, C_z, C_w)), dtype=np.float32)
        else:
            mu = np.array(list(zip(C_x, C_y)), dtype=np.float32)
        return mu, prev_classification
    elif initial_method == 'random-from-data':
        candidate = np.random.randint(low=0, high=data.shape[0], size=K)
        mu = np.zeros([K, K], dtype=np.float32)
        for i in range(K):
            mu[i, :] = data[candidate[i], :]
        return mu, prev_classification
    elif initial_method == 'Kmeans++':
        mu = np.zeros([K, K], dtype=np.float32)
        first_cluster = np.random.randint(low=0, high=data.shape[0], size=1, dtype=np.int)
        mu[0, :] = data[first_cluster, :]
        for i in range(1,K):
            distance = np.zeros(data.shape[0], dtype=np.float32)
            for j in range(0, data.shape[0]):
                distance[j] = np.linalg.norm(data[j, :] - mu[0, :])
            distance = distance / distance.sum()
            candidate = np.random.choice(data.shape[0], 1, p=distance)
            mu[i, :] = data[candidate, :]
        return mu, prev_classification
    
def classify(data, mu):
    classification = np.zeros(data.shape[0], dtype=np.int)
    for dataidx in range(data.shape[0]):
        distance = np.zeros(mu.shape[0], dtype=np.float)
        for cluster in range(mu.shape[0]):
            delta = abs(np.subtract(data[dataidx, :], mu[cluster, :]))
            distance[cluster] = np.square(delta).sum(axis=0)
        classification[dataidx] = np.argmin(distance)

    return classification

def calculate_error(classification, prev_classification):
    error = 0
    for i in range(classification.shape[0]):
        error += np.absolute(classification[i] - prev_classification[i])

    return error

def visualization(filename, storename, iteration, classification, initial_method):
    img = Image.open(filename)
    width, height = img.size
    pixel = img.load()
    color = [(0,0,0), (100, 0, 0), (0, 255, 0), (255,255,255)]
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = color[classification[i * num + j]]
    img.save(storename + '_' + initial_method + '_' + str(gamma_c) + '_' + str(gamma_s) + '_' + str(iteration) + '_'+ str(K) + '.png')

def draw_eigenspace(storename, classification, initial_method, data):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    plt.clf()
    title = "Spectral-Clustering in Eigen-Space"
    plt.suptitle(title)
    for cluster in range(K):
        col = next(color)
        for j in range(0, data.shape[0]):
            if classification[j] == cluster:
                plt.scatter(data[j][0], data[j][1], s=8, c=[col])
    plt.savefig(storename + '_' + initial_method + '_' + str(gamma_c) + '_' + str(gamma_s) + '_' + 'eigenspace' + '_'+ str(K) + '.png')

def update(data, mu, classification):
    new_mu = np.zeros(mu.shape, dtype=np.float32)
    count = np.zeros(mu.shape, dtype=np.int)
    one = np.ones(mu.shape[1], dtype=np.int)
    for dataidx in range(data.shape[0]):
        new_mu[classification[dataidx]] += data[dataidx]
        count[classification[dataidx]] += one
    for i in range(new_mu.shape[0]):
        if count[i][0] == 0:
            count[i] += one
    
    return np.true_divide(new_mu, count)

def K_Means(data, filename, storename):
    method = ['random', 'random-from-data', 'Kmeans++']
    for initial_method in method:
        print("Initial method: {}".format(initial_method))
        mu, classification = initial(data, initial_method)
        iteration = 0
        error = -10000
        prev_error = -10001
        print("mu = {}".format(mu))

        while(iteration <= epochs):
            iteration += 1
            print("iteration = {}".format(iteration))
            print("current mu = {}".format(mu))
            prev_classification = classification
            visualization(filename, storename, iteration, classification, initial_method)
            classification = classify(data, mu)
            error = calculate_error(classification, prev_classification)
            print("error = {}".format(error))
            if error == prev_error:
                break
            prev_error = error
            mu = update(data, mu, classification)
        
        draw_eigenspace(filename, storename, iteration, classification, initial_method, data)

def normalized_cut(pixel, coord):
    weight = compute_kernel(pixel, coord)
    degree = np.diag(np.sum(weight, axis=1))

    degree_square = np.diag(np.power(np.diag(degree), -0.5))
    L_sym = np.eye(weight.shape[0]) - degree_square @ weight @ degree_square
    eigen_values, eigen_vectors = np.linalg.eig(L_sym)
    idx = np.argsort(eigen_values)[1: K+1]
    U = eigen_vectors[:, idx].real.astype(np.float32)

    # normalized
    sum_over_row = (np.sum(np.power(U, 2), axis=1) ** 0.5).reshape(-1, 1)
    T = U.copy()
    for i in range(sum_over_row.shape[0]):
        if sum_over_row[i][0] == 0:
            sum_over_row[i][0] = 1
        T[i][0] /= sum_over_row[i][0]
        T[i][1] /= sum_over_row[i][0]
    
    return T

def ratio_cut(pixel, coord):
    weight = compute_kernel(pixel, coord)
    degree = np.diag(np.sum(weight, axis=1))
    L = degree - weight

    eigen_values, eigen_vectors = np.linalg.eig(L)
    idx = np.argsort(eigen_values)[1: K+1]
    U = eigen_vectors[:, idx].real.astype(np.float32)

    return U
    
if __name__ == '__main__':
    filename = 'data/image1.png'
    storename = 'visualization/image1_spectral_'
    pixel1, coord1 = read_input(filename)
    T = normalized_cut(pixel1, coord1)
    K_Means(T, filename, storename)

    filename = 'data/image2.png'
    storename = 'visualization/image2_spectral_'
    pixel2, coord2 = read_input(filename)
    T = normalized_cut(pixel2, coord2)
    K_Means(T, filename, storename)

    ###########################################################
    print("===================================================")

    filename = 'data/image1.png'
    storename = 'visualization/image1_spectral_ratio_'
    pixel1, coord1 = read_input(filename)
    U = ratio_cut(pixel1, coord1)
    K_Means(U, filename, storename)

    filename = 'data/image2.png'
    storename = 'visualization/image2_spectral_ratio_'
    pixel2, coord2 = read_input(filename)
    U = ratio_cut(pixel2, coord2)
    K_Means(U, filename, storename)
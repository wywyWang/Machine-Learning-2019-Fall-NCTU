import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def read_file():
    data_x = []
    data_y = []
    with open('./data/input.data') as file:
        for line in file:
            data_x.append(float(line.split()[0]))
            data_y.append(float(line.split()[1]))
    return np.array(data_x).reshape(-1, 1), np.array(data_y).reshape(-1, 1)

def kernel(X1, X2, params):
    scalelength = params[0]
    sigma = params[1]
    alpha = params[2]
    pairwise_sq_dists = (sigma ** 2) * ((cdist(X1, X2, 'sqeuclidean') / 2 * alpha * (scalelength ** 2)) + 1) ** (-alpha)
    return pairwise_sq_dists

def GPR_train(train_x, train_y, params):
    mu = np.zeros(train_x.shape)
    cov = kernel(train_x, train_x, params) + beta_inv * np.identity(train_x.shape[0])
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv

def GPR_test(train_x, train_y, params, test_num, mu, cov_inv):
    test_x = np.linspace(-60, 60, test_num).reshape(-1, 1)
    test_y = np.empty(test_num).reshape(-1, 1)
    test_y_plus = np.empty(test_num).reshape(-1, 1)
    test_y_minus = np.empty(test_num).reshape(-1, 1)
    
    k_test = kernel(test_x, test_x, params) + beta_inv
    k_train_test = kernel(train_x, test_x, params)
    test_y = np.linalg.multi_dot([k_train_test.T, cov_inv, train_y])
    std = np.sqrt(k_test - np.linalg.multi_dot([k_train_test.T, cov_inv, k_train_test]))
    test_y_plus = test_y + 2 * (np.diag(std).reshape(-1, 1))
    test_y_minus = test_y - 2 * (np.diag(std).reshape(-1, 1))

    return test_x, test_y, test_y_minus, test_y_plus

def NLL(params, train_x, train_y):
    k = kernel(train_x, train_x, params)
    sum1 = 0.5 * np.log(np.linalg.det(k))
    sum2 = 0.5 * np.linalg.multi_dot([train_y.T, np.linalg.inv(k), train_y])
    sum3 = 0.5 * train_x.shape[0] * np.log(2*np.pi)
    negative_log_likelihood = sum1 + sum2 + sum3
    return negative_log_likelihood[0][0]

def draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, params, mode):
    #Plot result
    fig = plt.figure()
    plt.title("Gaussian Process with scalelength : {0:.2f} sigma : {1:.2f} alpha : {2:.2f}".format(params[0], params[1], params[2]))
    plt.fill_between(test_x.ravel(), test_y_plus.ravel(), test_y_minus.ravel(), facecolor='pink')
    plt.scatter(train_x, train_y, color = 'black')
    plt.plot(test_x.ravel(), test_y.ravel(), color='blue')
    plt.plot(test_x.ravel(), test_y_plus.ravel(), color='red')
    plt.plot(test_x.ravel(), test_y_minus.ravel(), color='red')
    plt.xlim(-60, 60)
    fig.savefig("GP" + mode + ".png")

if __name__ == "__main__":
    train_x, train_y = read_file()
    params = [1.0, 1.0, 1.0]                         #initial guess
    test_num = 120
    beta_inv = 1/5

    mu, cov_inv = GPR_train(train_x, train_y, params)
    test_x, test_y, test_y_minus, test_y_plus = GPR_test(train_x, train_y, params, test_num, mu, cov_inv)
    draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, params, 'basic')

    #optimize hyperparameters
    fmin = minimize(fun=NLL, x0=params, args=(train_x, train_y), bounds=((1e-3, None), (1e-3, None), (1e-3, None)), method='L-BFGS-B', options={})
    print(fmin.x)

    #optimize parameter result
    mu, cov_inv = GPR_train(train_x, train_y, fmin.x)
    test_x, test_y, test_y_minus, test_y_plus = GPR_test(train_x, train_y, fmin.x, test_num, mu, cov_inv)
    draw(train_x, train_y, test_x, test_y, test_y_minus, test_y_plus, fmin.x, 'optimize')
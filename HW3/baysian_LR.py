import argparse
import numpy as np
import math
import matplotlib.pyplot as plt 

# py baysian_LR.py --B=1 --N=4 --S=1 --W 1 2 3 4

def univariate_generator(mean, variance):
    deviate = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return mean + deviate * math.sqrt(variance)

def polynomial_basis_linear_model(given_n, given_w, given_variance, x):
    y = 0.0
    for i in range(given_n):
        y += given_w[i] * (x**i)

    return np.array(y + univariate_generator(0, given_variance))

def build_design_matrix(n, x):
    A = []
    for i in range(n):
        A.append(x**i)

    return np.array(A).reshape(1, -1)

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=float)
    parser.add_argument("--N", type=int)
    parser.add_argument("--S", type=float)
    parser.add_argument("--W", nargs='+', type=float)
    args = parser.parse_args()

    given_n = args.N
    given_mean = 0
    given_variance = args.S
    given_w = args.W
    given_precision = args.B

    count = 1
    data_x = []
    data_y = []
    ten_data_x = []
    ten_data_y = []
    fifty_data_x = []
    fifty_data_y = []
    ten_posterior_mean = np.array([])
    fifty_posterior_mean = np.array([])
    ten_posterior_var_inv = np.array([])
    fifty_posterior_var_inv = np.array([])

    prior_mean = given_mean
    prior_var_inv = given_precision
    x = np.random.uniform(-1.0, 1.0)
    y = polynomial_basis_linear_model(given_n, given_w, given_variance, x)
    design_matrix = build_design_matrix(given_n, x)
    data_x.append(x)
    data_y.append(y)

    #Fist iteration
    posterior_var_inv = given_variance * np.matmul(design_matrix.T,design_matrix) + prior_var_inv * np.eye(given_n)
    posterior_mean = given_variance * np.matmul(np.linalg.inv(posterior_var_inv), design_matrix.T) * y
    predictive_distribution_mean = np.matmul(design_matrix, posterior_mean)
    predictive_distribution_variance = 1 / given_variance + np.matmul(np.matmul(design_matrix, np.linalg.inv(posterior_var_inv)), design_matrix.T)

    print("Add data point ({}, {}):".format(x, y))
    print("")
    print("Posterior mean:")
    print(posterior_mean)
    print("")
    print("Posterior covariance:")
    print(np.linalg.inv(posterior_var_inv))
    print("")
    print("Predictive distribution ~ N({}, {})".format(predictive_distribution_mean, predictive_distribution_variance))
    print("--------------------------------------------------")

    while True:
        count += 1
        x = np.random.uniform(-1.0, 1.0)
        y = polynomial_basis_linear_model(given_n, given_w, given_variance, x)
        data_x.append(x)
        data_y.append(y)

        design_matrix = build_design_matrix(given_n, x)
        prior_mean = posterior_mean.copy()
        prior_var_inv = posterior_var_inv.copy()

        posterior_var_inv = given_variance * np.matmul(design_matrix.T, design_matrix) + prior_var_inv
        posterior_mean = np.matmul(np.linalg.inv(posterior_var_inv), (given_variance * design_matrix.T * y + np.matmul(prior_var_inv, prior_mean)))
        predictive_distribution_mean = np.matmul(design_matrix, posterior_mean)
        predictive_distribution_variance = 1 / given_variance + np.matmul(np.matmul(design_matrix, np.linalg.inv(posterior_var_inv)), design_matrix.T)

        print("Count = {}".format(count))
        print("Add data point ({}, {}):".format(x, y))
        print("")
        print("Posterior mean:")
        print(posterior_mean)
        print("")
        print("Posterior covariance:")
        print(np.linalg.inv(posterior_var_inv))
        print("")
        print("Predictive distribution ~ N({}, {})".format(predictive_distribution_mean, predictive_distribution_variance))
        print("--------------------------------------------------")

        if (abs(np.sum(prior_mean - posterior_mean)) < 1e-6) and (abs(np.sum(np.linalg.inv(prior_var_inv) - np.linalg.inv(posterior_var_inv))) < 1e-6):
            break

        # #Testing
        # if count == 3:
        #     break

        if count == 10:
            ten_posterior_mean = posterior_mean.copy()
            ten_posterior_var_inv = posterior_var_inv.copy()
            ten_data_x = data_x.copy()
            ten_data_y = data_y.copy()

        if count == 50:
            fifty_posterior_mean = posterior_mean.copy()
            fifty_posterior_var_inv = posterior_var_inv.copy()
            fifty_data_x = data_x.copy()
            fifty_data_y = data_y.copy()

    #ground truth
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title("Ground Truth")
    ground_func = np.poly1d(np.flip(given_w))
    ground_x = np.linspace(-2.0, 2.0, 30)
    ground_y = ground_func(ground_x)
    plt.plot(ground_x, ground_y, color = 'black')
    ground_y += given_variance                          #mean + variance
    plt.plot(ground_x, ground_y, color = 'red')
    ground_y -= 2 * given_variance                      #mean + variance - 2 * variance
    plt.plot(ground_x, ground_y, color = 'red')
    fig.savefig('Ground truth.png')

    #predict result
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title("Predict result")
    predict_x = np.linspace(-2.0, 2.0, 30)
    predict_func = np.poly1d(np.flip(posterior_mean.flatten()))
    predict_y = predict_func(predict_x)
    predict_y_plus = predict_func(predict_x)
    predict_y_minus = predict_func(predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = build_design_matrix(given_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / given_variance + np.matmul(np.matmul(predict_design_matrix, np.linalg.inv(posterior_var_inv)), predict_design_matrix.T)
        predict_y_plus[i] += predict_predictive_distribution_variance[0]
        predict_y_minus[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color='black')
    plt.plot(predict_x, predict_y_plus, color='red')
    plt.plot(predict_x, predict_y_minus, color='red')
    plt.scatter(data_x, data_y)
    fig.savefig('Predict result.png')

    #10 incomes
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title("After 10 incomes")
    predict_func = np.poly1d(np.flip(ten_posterior_mean.flatten()))
    predict_y = predict_func(predict_x)
    predict_y_plus = predict_func(predict_x)
    predict_y_minus = predict_func(predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = build_design_matrix(given_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / given_variance + np.matmul(np.matmul(predict_design_matrix, np.linalg.inv(ten_posterior_var_inv)), predict_design_matrix.T)
        predict_y_plus[i] += predict_predictive_distribution_variance[0]
        predict_y_minus[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color='black')
    plt.plot(predict_x, predict_y_plus, color='red')
    plt.plot(predict_x, predict_y_minus, color='red')
    plt.scatter(ten_data_x, ten_data_y)
    fig.savefig('After 10 imcomes.png')


    #50 incomes
    fig = plt.figure()
    plt.xlim(-2.0, 2.0)
    plt.title("After 50 incomes")
    predict_func = np.poly1d(np.flip(fifty_posterior_mean.flatten()))
    predict_y = predict_func(predict_x)
    predict_y_plus = predict_func(predict_x)
    predict_y_minus = predict_func(predict_x)

    for i in range(len(predict_x)):
        predict_design_matrix = build_design_matrix(given_n,predict_x[i])
        predict_predictive_distribution_variance = 1 / given_variance + np.matmul(np.matmul(predict_design_matrix, np.linalg.inv(fifty_posterior_var_inv)), predict_design_matrix.T)
        predict_y_plus[i] += predict_predictive_distribution_variance[0]
        predict_y_minus[i] -= predict_predictive_distribution_variance[0]

    plt.plot(predict_x, predict_y, color='black')
    plt.plot(predict_x, predict_y_plus, color='red')
    plt.plot(predict_x, predict_y_minus, color='red')
    plt.scatter(fifty_data_x, fifty_data_y)
    fig.savefig('After 50 imcomes.png')


    print("ten mean = {}".format(np.flip(ten_posterior_mean.flatten())))
    print("ten variance = {}".format(ten_posterior_var_inv))
    print("fifty mean = {}".format(np.flip(fifty_posterior_mean.flatten())))
    print("fifty variance = {}".format(fifty_posterior_var_inv))
    print("final mean = {}".format(np.flip(posterior_mean.flatten())))
    print("final variance = {}".format(posterior_var_inv))
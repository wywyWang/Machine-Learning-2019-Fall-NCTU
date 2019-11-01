import argparse
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=5, suppress=True)

# py logistic_regression.py --N=50 --MX1=1 --MY1=1 --MX2=10 --MY2=10 --VX1=2 --VY1=2 --VX2=2 --VY2=2
# py logistic_regression.py --N=50 --MX1=1 --MY1=1 --MX2=3 --MY2=3 --VX1=2 --VY1=2 --VX2=4 --VY2=4

def univariate_generator(mean,variance):
    deviate = np.sum(np.random.uniform(0,1,12)) - 6
    return mean + deviate * math.sqrt(variance)

def sigmoid(z):
    result = []
    for i in range(np.shape(z)[0]):
        sigmoid_result = 1 / (1 + np.exp(-z[i]))
        result.append(sigmoid_result)

    return np.array(result)

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type = int  )
    parser.add_argument("--MX1", type = float  )
    parser.add_argument("--VX1", type = float  )
    parser.add_argument("--MY1", type = float  )
    parser.add_argument("--VY1", type = float  )
    parser.add_argument("--MX2", type = float  )
    parser.add_argument("--VX2", type = float  )
    parser.add_argument("--MY2", type = float  )
    parser.add_argument("--VY2", type = float  )
    args = parser.parse_args()
    given_N = args.N
    given_MX1 = args.MX1
    given_VX1 = args.VX1
    given_MY1 = args.MY1
    given_VY1 = args.VY1
    given_MX2 = args.MX2
    given_VX2 = args.VX2
    given_MY2 = args.MY2
    given_VY2 = args.VY2

    D1 = []
    D2 = []
    D1y = []
    D2y = []
    design_matrix = []
    y = []

    for i in range(given_N):
        dx1 = univariate_generator(given_MX1,given_VX1)
        dy1 = univariate_generator(given_MY1,given_VY1)
        design_matrix.append([1, dx1, dy1])
        y.append(0)
        D1.append([dx1, dy1])
        D1y.append(0)

    for i in range(given_N):
        dx2 = univariate_generator(given_MX2,given_VX2)
        dy2 = univariate_generator(given_MY2,given_VY2)
        design_matrix.append([1, dx2, dy2])
        y.append(1)
        D2.append([dx2, dy2])
        D2y.append(1)
        
    design_matrix = np.array(design_matrix)
    y = np.array(y).reshape(-1, 1)
    D1 = np.array(D1)
    D2 = np.array(D2)
    D1y = np.array(D1y)
    D2y = np.array(D2y)

    weight = np.random.randn(design_matrix.shape[1], 1).reshape(-1, 1)

    print("design_matrix shape = {}".format(np.shape(design_matrix)))
    print("y shape = {}".format(np.shape(y)))
    print("weight shape = {}".format(np.shape(weight)))

    #Gradient Descent
    epochs = int(1e2)
    learning_rate = 0.95
    gradient_weight = weight.copy()
    for count in range(epochs):
        gradient = np.subtract(y, sigmoid(np.matmul(design_matrix, gradient_weight)))
        gradient_weight = gradient_weight + learning_rate * np.matmul(design_matrix.T, gradient)


    print("gradient weight = {}".format(gradient_weight))

    predict = np.matmul(design_matrix, gradient_weight)
    
    #For plotting
    gdD1 = []
    gdD2 = []
    gdD1y = []
    gdD2y = []

    for row in range(np.shape(predict)[0]):
        if predict[row][0] <= 0.5:
            gdD1.append([design_matrix[row][1], design_matrix[row][2]])
        else:
            gdD2.append([design_matrix[row][1], design_matrix[row][2]])

    gdD1 = np.array(gdD1)
    gdD2 = np.array(gdD2)
        
    predict[predict > 0.5] = 1
    predict[predict <= 0.5] = 0

    #Counfusion matrix
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for row in range(np.shape(predict)[0]):
        if predict[row][0] == 0:
            if y[row][0] == 1:
                FN += 1
            elif y[row][0] == 0:
                TN += 1
        elif predict[row][0] == 1:
            if y[row][0] == 1:
                TP += 1
            elif y[row][0] == 0:
                FP += 1

    print("Confusion matrix:")
    print("\t\tPredict cluster 0\tPredict cluster 1")
    print("Real cluster 0\t\t{}\t\t\t{}".format(TN,FP))
    print("Real cluster 1\t\t{}\t\t\t{}".format(FN,TP))
    print("Sensitivity (Successfully predict cluster 0): {}".format(TN / (TN + FP)))
    print("Specificity (Successfully predict cluster 1): {}".format(TP / (TP + FN)))


    fig = plt.figure()
    plt.title("Ground Truth")
    plt.scatter(D1[:,0], D1[:,1], c = 'r')
    plt.scatter(D2[:,0], D2[:,1], c = 'b')
    fig.savefig('ground truth.png')

    fig = plt.figure()
    plt.title("Gradient Descent")
    plt.scatter(gdD1[:,0], gdD1[:,1], c = 'r')
    plt.scatter(gdD2[:,0], gdD2[:,1], c = 'b')
    fig.savefig('gradient descent.png')
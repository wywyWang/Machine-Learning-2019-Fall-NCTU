import argparse
import numpy as np
import math

# py sequential_estimator.py --M=3.0 --S=5.0

def univariate_generator(mean,variance):
    deviate = np.sum(np.random.uniform(0,1,12)) - 6
    return mean + deviate * math.sqrt(variance)

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type = float  )
    parser.add_argument("--S", type = float  )
    args = parser.parse_args()
    given_mean = args.M
    given_variance = args.S

    est_mean = 1e6
    est_variance = 1e6
    #first point
    pre_mean = univariate_generator(given_mean, given_variance)
    pre_variance = 0
    count = 1

    while True:
        point = univariate_generator(given_mean, given_variance)
        count += 1
        est_mean = ((count - 1) * pre_mean + point) / count
        est_variance = pre_variance + ((point - pre_mean) * (point - est_mean) - pre_variance) / count

        print("Count = {}".format(count))
        print("Add data point: {}".format(point))
        print("Mean = {}\t Variance = {}".format(est_mean, est_variance))
        print("")

        if (abs(est_mean - given_mean) < 1e-3 and abs(est_variance - given_variance) < 1e-3):
            break
        else:
            pre_mean = est_mean
            pre_variance = est_variance
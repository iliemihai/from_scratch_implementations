import cPickle
import gzip

import numpy as np

def load_data(file_path):
    f = gzip.open(file_path, "rb")
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def vectorized_results(i):
    n = np.zeros((10, 1))
    n[i] = 1.0
    return n

def load_data_wrapper():
    training_data, validation_data, test_data = load_data("./mnist.pkl.gz")
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_targets = [vectorized_results(y) for y in training_data[1]]
    training_final = zip(training_inputs, training_targets)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_final = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_final = zip(test_inputs, test_data[1])
    return (training_final, validatio_final, test_final)

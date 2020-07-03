#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import sys


def load_data():
    '''
    Function to read data either from file location or as command line arguments and manipulate them to appropriate format for matrix multiplication in later steps
    :return: x_train, x_test, y_train
    '''
    args = sys.argv
    # Read as command line arguments
    if args:
        train_image_file = args[1]
        train_label_file = args[2]
        test_image_file = args[3]
    # Read from file
    else:
        train_image_file = 'train_image.csv'
        train_label_file = 'train_label.csv'
        test_image_file = 'test_image.csv'
    # Load data
    x_train = np.loadtxt(train_image_file, delimiter=',')
    y_train = np.loadtxt(train_label_file, delimiter=',')
    x_test = np.loadtxt(test_image_file, delimiter=',')
    # One hot encode the train labels for machine parsing
    y_train = y_train.astype(int)
    y_train_vector = [one_hot_encode_labels(y) for y in y_train]
    y_train = np.array(y_train_vector).reshape(len(x_train), 10)
    # Normalize x by dividing by pixel count
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.
    return x_train, x_test, y_train


def one_hot_encode_labels(l):
    '''
    Function to one hot encode data for machine parsing
    :param l: data to be encoded i.e. single row of train label data
    :return: one hot encoded data in 1*10 vector format
    '''
    labels = np.zeros((10, 1))
    labels[l] = 1.0
    return labels.transpose()


def sigmoid(z):
    '''
    Activation function to squishify the data to values between 0 and 1
    :param z: Numpy array
    :return: Numpy array with values between 0 and 1
    '''
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''
    Function to calculate the derivative of the sigmoid function
    :param z: Activated numpy array
    :return: Numpy array with sigmoid derivatives of each value
    '''
    return z * (1 - z)


class MNITSNetwork(object):

    def __init__(self, sizes):
        '''
        Function to initialize vectors of weights and biases, number and size of layers
        :param sizes: List of number and size of layers
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        num_neurons_input_layer = self.sizes[0]
        num_neurons_first_layer = self.sizes[1]
        num_neurons_second_layer = self.sizes[2]
        num_neurons_output_layer = self.sizes[3]
        # Initialize weight matrices with Xavier initialization
        # Initialize weight matrix with random weights of size (number of input neurons) x (number of first layer neurons)
        self.weight_matrix1 = np.random.normal(loc=0.0,
                                               scale=np.sqrt(2 / (num_neurons_input_layer + num_neurons_first_layer)),
                                               size=(num_neurons_input_layer, num_neurons_first_layer))
        # Initialize random matrix of size (number of first layer neurons) x (number of second layer neurons)
        self.weight_matrix2 = np.random.normal(loc=0.0,
                                               scale=np.sqrt(2 / (num_neurons_first_layer + num_neurons_second_layer)),
                                               size=(num_neurons_first_layer, num_neurons_second_layer))
        # Initialize random matrix of size (number of first layer neurons) x (number of second layer neurons)
        self.weight_matrix3 = np.random.normal(loc=0.0,
                                               scale=np.sqrt(2 / (num_neurons_second_layer + num_neurons_output_layer)),
                                               size=(num_neurons_second_layer, num_neurons_output_layer))
        # Initialize bias matrices with zero vector biases of size based on number neurons in hidden and output layers
        self.bias_matrix1 = np.zeros((1, num_neurons_first_layer))
        self.bias_matrix2 = np.zeros((1, num_neurons_second_layer))
        self.bias_matrix3 = np.zeros((1, num_neurons_output_layer))

    def cross_entropy(self, predictions, actuals):
        '''
        Function to calculate cross entropy between predicted and actual values
        :param predictions: Vector output of forward propagation in neural network
        :param actuals: Vector input from training labels
        :return: Normalized vector describing errors in actual and predicted values
        '''
        normalization_factor = actuals.shape[0]
        # residual = predictions - actuals
        return (predictions - actuals) / normalization_factor

    def backpropagation(self, x_train, y_train, learning_rate):
        '''
        Function to backpropagate the errors and adjust weights and biases in order to minimize error and train the network
        :param x_train: Vector of handwritten digit pixel data
        :param y_train: Vector of one-hot encoded train labels
        :param learning_rate: Hyperparameter to define rate of change in weights
        '''
        # Cross entropy to calculate errors between predicted values (from forward propagation) and actual values
        hidden_layer_2_output_layer_delta = self.cross_entropy(self.output_layer_activation, y_train)

        # Propagate error in output layer to hidden layer 2 based on weights of hidden layer 2 and output layer
        hidden_layer_1_hidden_layer_2_delta = np.dot(hidden_layer_2_output_layer_delta, self.weight_matrix3.transpose())
        hidden_layer_1_hidden_layer_2_activation_delta = hidden_layer_1_hidden_layer_2_delta * sigmoid_prime(
            self.hidden_layer_2_activation)

        # Propagate error in output layer to hidden layer 1 based on weights of hidden layer 2 and 1
        input_layer_hidden_layer_1_delta = np.dot(hidden_layer_1_hidden_layer_2_activation_delta, self.weight_matrix2.transpose())
        input_layer_hidden_layer_1_activation_delta = input_layer_hidden_layer_1_delta * sigmoid_prime(
            self.hidden_layer_1_activation)

        # Adjust weights to minimize errors in predicted values
        self.weight_matrix3 -= learning_rate * np.dot(self.hidden_layer_2_activation.transpose(),
                                                      hidden_layer_2_output_layer_delta)
        self.weight_matrix2 -= learning_rate * np.dot(self.hidden_layer_1_activation.transpose(),
                                                      hidden_layer_1_hidden_layer_2_activation_delta)
        self.weight_matrix1 -= learning_rate * np.dot(x_train.transpose(), input_layer_hidden_layer_1_activation_delta)

        # Adjust biases to minimize errors in predicted values
        self.bias_matrix3 -= learning_rate * np.sum(hidden_layer_2_output_layer_delta, axis=0, keepdims=True)
        self.bias_matrix2 -= learning_rate * np.sum(hidden_layer_1_hidden_layer_2_activation_delta, axis=0)
        self.bias_matrix1 -= learning_rate * np.sum(input_layer_hidden_layer_1_activation_delta, axis=0)

    def feedforward(self, input_matrix):
        '''
        Function to train the neural network to make predictions on supplied data
        :param input_matrix: Vector of training data
        '''
        # Input layer - Hidden Layer 1
        input_layer_hidden_layer_1_z = np.dot(input_matrix, self.weight_matrix1) + self.bias_matrix1
        # Sigmoid activation on hidden layer 1 to convert to accommodate non-linearity of input function
        self.hidden_layer_1_activation = sigmoid(input_layer_hidden_layer_1_z)

        # Hidden Layer 1 - Hidden Layer 2
        hidden_layer_1_hidden_layer_2_z = np.dot(self.hidden_layer_1_activation,
                                                 self.weight_matrix2) + self.bias_matrix2
        # Sigmoid activation on hidden layer 2 to convert to accommodate non-linearity of input function
        self.hidden_layer_2_activation = sigmoid(hidden_layer_1_hidden_layer_2_z)

        # Hidden Layer 2 - Output Layer
        hidden_layer_2_output_layer_2_z = np.dot(self.hidden_layer_2_activation,
                                                 self.weight_matrix3) + self.bias_matrix3
        # Softmax activation on output layer to convert output vector into a classification probability vector
        self.output_layer_activation = self.softmax_activation(hidden_layer_2_output_layer_2_z)

    def get_preds(self, test_data):
        '''
        Get predictions on test data by forward propagating it through the trained neural network
        :param test_data: Vector of test data containing pixels of handwritten digit images
        :return: List of predictions
        '''
        # Initialize list to contain labels of predictions for current epoch
        epoch_preds = []
        # Iterate through each row of test data
        for val in test_data:
            # Forward propagate the data
            self.feedforward(val)
            # Get label of most probable classification
            s = self.output_layer_activation.argmax()
            # Append label to list
            epoch_preds.append(s)
        # Return predictions for current epoch
        return epoch_preds

    def train(self, num_epochs, learning_rate, batch_size, x_train, y_train, x_test):
        '''
        Function to train the neural network using stochastic gradient descent approach
        :param num_epochs: Number of training epochs
        :param learning_rate: Learning rate of neural network
        :param batch_size: Size of each batch of training data
        :param x_train: Training data vector
        :param y_train: Testing data vector
        :param x_test: Testing data vector
        :return: Prediction labels
        '''
        # Get length of training data
        n = len(x_train)
        # List to hold prediction labels for test data
        preds = []
        # Iterate over each epoch
        for epoch in range(num_epochs):
            # Shuffle training data and corresponding labels
            random_indices = np.random.permutation(n)
            x_train = x_train[random_indices]
            y_train = y_train[random_indices]
            # For each batch of training data
            for i in range(0, n, batch_size):
                # Create batch data
                x_train_mini = x_train[i:i + batch_size]
                y_train_mini = y_train[i:i + batch_size]
                # Feedforward data to get predictions
                self.feedforward(x_train_mini)
                # Backpropagate errors to update weights and biases
                self.backpropagation(x_train_mini, y_train_mini, learning_rate)
            # Append predictions of current epoch to list
            preds.append(self.get_preds(x_test))
        return preds

    def softmax_activation(self, s):
        '''
        Function to convert the output of neural network into probabilities between 0 and 1 to give a degree of belief to classifications
        :param s: Vector from output layer
        :return: Vector of classification probabilities
        '''
        return np.exp(s - np.max(s, axis=1, keepdims=True))/ np.sum(np.exp(s - np.max(s, axis=1, keepdims=True)), axis=1, keepdims=True)


# Load data
x_train, x_test, y_train = load_data()
# Initialize class of neural network with required number of neurons in each layer
net = MNITSNetwork([784, 128, 128, 10])
# Get predictions on test data using the trained neural network
preds = net.train(70, 0.5, 50, x_train, y_train, x_test)
# Save predictions to file
np.savetxt("test_predictions.csv", preds[-1], fmt='%i', delimiter=",")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# activiation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_prime(x):
    return x * (1.0 - x)

# mean squared Error 
def mse(target,output):
    return np.average((target - output) ** 2)

def accuracy(pred, actual):
    num_correct = 0
    for i in range(len(pred)):
        a = pred[i]
        b = actual[i]
        if a == b:
            num_correct += 1
    return num_correct / len(pred)

def encode_target_to_numeric(df):
    targets = ['ascendingStairs', 'basketBall', 'crossTrainer','cyclingHorizontal', 'cyclingVertical', 'decendingStairs','jumping', 'lyingBack', 'lyingRigh', 'movingInElevator', 'rowing','runningTreadmill', 'sitting', 'standing','standingInElevatorStill', 'stepper', 'walkingLot','walkingTreadmillFlat', 'walkingTreadmillIncline']
    encodings = range(1,20)
    for i in range(len(targets)):
        df['activity'] = df['activity'].replace([targets[i]], encodings[i])
    return df

# Transforms target for the mse measurement
def transform_target(targets_org):
    output_size = len(np.unique(targets_org))
    y = np.zeros((len(targets_org),output_size))
    i = 0
    for cur in y:
        index = int(targets_org[i] - 1)
        index = int(index - 1)
        cur[index] = 1
        i = i + 1
    return y


class Neural_Net:
    def __init__(self, nodes_per_layer, num_inputs, num_outputs):
        # number of nodes in each hidden layer
        self.nodes_per_layer = nodes_per_layer
        # list containing weights for all connections
        self.weights = []
        # list containing activation at each point in network
        self.activations = []
        # list containing dir for gradiant desent
        self.derivatives = []
        # num inputs
        self.num_inputs = num_inputs
        # number of outputs
        self.num_outputs = num_outputs
        # initialize activations, derivatives, and weights
        self.init_lists(num_inputs, num_outputs)


    def init_lists(self, num_inputs, num_outputs):
        network = [num_inputs] + self.nodes_per_layer + [num_outputs]

        # init weights and derivativs lists
        for i in range(len(network) - 1):
            w = np.random.rand(network[i], network[i + 1])   
            d = np.zeros((network[i], network[i + 1]))    
            self.weights.append(w)
            self.derivatives.append(d)

        # init activations list
        for i in range(len(network)):
            a = np.zeros(network[i])
            self.activations.append(a)

    def fit(self, X, y, num_iter, alpha):
        all_errors = []
        for i in range(num_iter):
            err_sum = 0

            exp_index = 0
            for training_exp in X:
                # predictions for training example x
                actual = self.forward_propagate(training_exp)
                # true value for training example
                expected = y[exp_index]

                error = expected - actual

                self.back_propagate(error)

                self.gradient_descent(alpha)

                err_sum += mse(expected, actual)
                exp_index += 1
            avg_error = err_sum / len(X)
            all_errors.append(avg_error)
            # Epoch complete, report the training error
            print("Avg error: {} at epoch {}".format(avg_error, i+1))
        return all_errors

    def forward_propagate(self, x):
        # initial activation is the input in the begining
        activations = x

        self.activations[0] = activations

        i = 1
        for cur_layer_weights in self.weights:
            # linerear combination of weights and activations
            z = np.dot(activations, cur_layer_weights)
            activations = sigmoid(z)
            self.activations[i] = activations
            i += 1

        return activations

    def predict(self,X):
        predictions = []
        for x in X:
            output = self.forward_propagate(x)
            pred = np.where(output == np.max(output))[0] + 2
            predictions.append(pred)
        pred = []
        for i in predictions:
            index = i[0]
            if index <= 19:
                pred.append(index)
            else:
                pred.append(index % 19)
        return pred

    """ ######################################################################################
    https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/8-%20Training%20a%20neural%20network:%20Implementing%20back%20propagation%20from%20scratch/code
    """ ######################################################################################

    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * sigmoid_prime(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)




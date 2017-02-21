import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def debug(expression):
    frame = sys._getframe(1)
    value = str(eval(expression, frame.f_globals, frame.f_locals))
    if '\n' in value:
        delimiter = ':\n'
    else:
        delimiter = ' ='
    print(expression + delimiter, value)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.layer_0_nodes = input_nodes
        self.layer_1_nodes = hidden_nodes
        self.layer_2_nodes = output_nodes

        # Seed
        np.random.seed(1)

        # Init weights
        self.weights_0_1 = np.random.normal(0.0, self.layer_1_nodes**-0.5, (self.layer_0_nodes, self.layer_1_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.layer_2_nodes**-0.5, (self.layer_1_nodes, self.layer_2_nodes))

        self.learn_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_output_2_derivative(output):
        return output * (1 - output)

    def train(self, inputs, targets, iterations):
        assert(len(inputs) == len(targets))
        for iteration in range(iterations):

            # self.weights_0_1.shape # 3x4
            # self.weights_1_2.shape # 4x1

            # inputs.shape # 6x3
            # targets.shape # 6x1

            # Forward Pass #
            hidden_input = inputs.dot(self.weights_0_1)  # 6x4
            hidden_output = self.sigmoid(hidden_input)  # 6x4

            final_input = hidden_output.dot(self.weights_1_2)  # 6x1
            final_output = self.sigmoid(final_input)  # 6x1

            # Back Pass #
            final_error = targets - final_output  # 6x1
            final_grad = final_error * self.sigmoid_output_2_derivative(final_output)  # 6x1

            hidden_error = final_error.dot(self.weights_1_2.T)  # 6x4
            hidden_grad = hidden_error * self.sigmoid_output_2_derivative(hidden_output)  # 6x4

            # Update Weights #
            self.weights_1_2 += hidden_output.T.dot(final_grad) * self.learn_rate
            self.weights_0_1 += inputs.T.dot(hidden_grad) * self.learn_rate

    def predict(self, inputs):
        # Forward Pass #
        hidden_input = inputs.dot(self.weights_0_1)  # 6x4
        hidden_output = self.sigmoid(hidden_input)  # 6x4

        final_input = hidden_output.dot(self.weights_1_2)  # 6x1
        final_output = self.sigmoid(final_input)  # 6x1
        return final_output


if __name__ == "__main__":
    network = NeuralNetwork(3, 4, 1, 0.01)

    features = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    labels   = np.array([[1,         1,         0,         0,         0,         0,         1,         1]]).T

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.25, random_state=42)

    network.train(features_train, labels_train, 100000)

    debug('features_test')
    debug('network.predict(features_test)')
    debug('labels_test')

    accuracy = accuracy_score(labels_test, np.round(network.predict(features_test)))

    debug('accuracy')

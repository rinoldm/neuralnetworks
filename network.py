import util
import functions as f
import draw

import random
from datetime import datetime
import numpy as np

def one_hot(size, input):
    one_hot_vector = np.zeros(size)
    one_hot_vector[input] = 1
    return one_hot_vector
        
class Layer:
    def __init__(self, size, activation, act_params):
        self.weights = [[] for _ in range(size)]
        self.biases = np.zeros(size)
        self.outputs = np.zeros(size)
        self.errors = np.zeros(size)
        if activation not in f.activations:
            raise ValueError("Activation function \"{}\" doesn't exist (available: {})".format(activation, list(f.activations.keys())))
        self.activation = activation
        self.act_params = act_params

class Network:
    def __init__(self):
        self.layers = []
        self.run_start = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.loss = ""
        self.regularization = ""
        self.reg_lambda = 0
        self.draw = None
        
    def __str__(self):
        s = ""
        for layer_index, layer in enumerate(self.layers):
            s += str(len(layer.outputs)) + "-" + layer.activation
            if layer_index < len(self.layers) - 1:
                s += "|"
        return s
    
    def add_layer(self, size, activation, act_params=[]):
        self.layers.append(Layer(size, activation, act_params))

    def initialize_weights(self):
        for layer_index, layer in enumerate(self.layers[1:], start=1):
            n_in = len(self.layers[layer_index - 1].outputs)
            if layer_index < len(self.layers) - 1:
                n_out = len(self.layers[layer_index + 1].outputs)
                if f.activations[layer.activation]["relu_family"]:
                    layer.weights = f.initializations["he_uniform"](n_in, n_out, len(layer.outputs))
                else:
                    layer.weights = f.initializations["xavier_uniform"](n_in, n_out, len(layer.outputs))
            else:
                layer.weights = f.initializations["output"](n_in, n_out, len(layer.outputs))

    def feed_forward(self, example_image):
        self.layers[0].outputs = example_image.flatten() / 255
        for layer_index, layer in enumerate(self.layers[1:], start=1):
            inputs = self.layers[layer_index - 1].outputs
            layer.outputs = f.activations[layer.activation]["func"](np.dot(layer.weights, inputs) + layer.biases, layer.act_params)
    
    def backpropagate_error(self, expected_outputs):
        output_layer = self.layers[-1]
        if output_layer.activation == "softmax" and self.loss == "cross_entropy":
            output_layer.errors = output_layer.outputs - expected_outputs
        else:
            loss_derivative = f.losses[self.loss]["deri"](output_layer.outputs, expected_outputs)
            activation_derivative = f.activations[output_layer.activation]["deri"](output_layer.outputs)
            if output_layer.activation == "softmax":
                output_layer.errors = np.dot(loss_derivative, activation_derivative)
            else:
                output_layer.errors = loss_derivative * activation_derivative

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].errors = np.dot(self.layers[i + 1].weights.T, self.layers[i + 1].errors)

    def train_batch(self, training_data_images, training_data_labels, batch_start, batch_end, learning_rate):
        if (batch_start == batch_end):
            return 0

        batch_gradients = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        batch_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        batch_loss = 0
        for example_index in range(batch_start, batch_end):
            self.feed_forward(training_data_images[example_index])
            expected_outputs = one_hot(len(self.layers[-1].outputs), training_data_labels[example_index])
            batch_loss += f.losses[self.loss]["func"](self.layers[-1].outputs, expected_outputs)
            
            self.backpropagate_error(expected_outputs)
            for layer_index, layer in enumerate(self.layers[1:], start=1):
                inputs = np.atleast_2d(self.layers[layer_index - 1].outputs)
                batch_gradients[layer_index - 1] += np.dot(np.atleast_2d(layer.errors).T, inputs)
                batch_biases[layer_index - 1] += layer.errors

        for layer_index, layer in enumerate(self.layers[1:], start=1):
            batch_loss += f.regularizations[self.regularization]["func"](layer.weights, self.reg_lambda)
            batch_gradients[layer_index - 1] += f.regularizations[self.regularization]["deri"](layer.weights, self.reg_lambda)
            #layer.weights -= np.clip((learning_rate * batch_gradients[layer_index - 1]) / (batch_end - batch_start), -1, 1)
            layer.weights -= learning_rate * batch_gradients[layer_index - 1] / (batch_end - batch_start)        
            layer.biases -= learning_rate * batch_biases[layer_index - 1] / (batch_end - batch_start)

        return batch_loss
    
    def train(self, images, labels, TRAINING, VALIDATING, epochs, learning_rate, batch_size=32, loss="mean_squared_error", regularization="none", reg_lambda=1e-3):
        if not "deri" in f.activations[self.layers[-1].activation]:
            raise ValueError("Activation function \"{}\" not suitable for output layer (no derivative implemented)".format(self.layers[-1].activation))
        if loss not in f.losses:
            raise ValueError("Loss function \"{}\" doesn't exist (available: {})".format(loss, list(f.losses.keys())))
        if regularization not in f.regularizations:
            raise ValueError("Regularization function \"{}\" doesn't exist (available: {})".format(regularization, list(f.regularizations.keys())))

        training_size, validating_size = TRAINING[1] - TRAINING[0], VALIDATING[1] - VALIDATING[0]
        self.loss, self.regularization, self.reg_lambda = loss, regularization, reg_lambda
        self.draw.loss_init(self)
        print("TRAINING: examples={}, batch_size={}, epochs={}, learning_rate={}, loss={}{}".format(training_size, batch_size, epochs, learning_rate, loss, ", regularization=" + regularization + ", reg_lambda=" + str(reg_lambda) if regularization != "none" else ""))
        time_start = datetime.now()
        for epoch in range(epochs):
            training_loss = 0
            for batch_index in range(training_size // batch_size):
                training_loss += self.train_batch(images, labels, batch_index * batch_size, (batch_index + 1) * batch_size, learning_rate)
            training_loss += self.train_batch(images, labels, (batch_index + 1) * batch_size, (batch_index + 1) * batch_size + training_size % batch_size, learning_rate)
            training_loss /= training_size
            validation_loss, correct_nb = self.validate(images, labels, VALIDATING)
            correct_rate = 100 * correct_nb / validating_size
            print("epoch {}: t_loss={}, v_loss={}, validation={}% ({}/{})".format(epoch, '%.6f' % training_loss, '%.6f' % validation_loss, '%.2f' % correct_rate, correct_nb, validating_size))
            self.draw.t_loss_history.append(training_loss)
            self.draw.v_loss_history.append(validation_loss)
            self.draw.val_history.append(correct_rate)
            self.draw.network_update(self)
            self.draw.loss_update(self)
            if (len(self.draw.v_loss_history) > 10 and self.draw.v_loss_history[-1] > self.draw.v_loss_history[-2]):
                print("early stopping (validation loss reached a minimum)")
                break
            if epoch % 10 == 0:
                util.save_weights(self)
        util.save_weights(self)
        print("TRAINING finished in {}".format(datetime.now() - time_start))

    def validate(self, images, labels, VALIDATING):
        total_validation_loss, correct_nb = 0, 0
        for example_index in range(VALIDATING[0], VALIDATING[1]):
            self.feed_forward(images[example_index])
            expected_class = labels[example_index]
            expected_outputs = one_hot(len(self.layers[-1].outputs), expected_class)
            total_validation_loss += f.losses[self.loss]["func"](self.layers[-1].outputs, expected_outputs)
            
            predicted_class = np.argmax(self.layers[-1].outputs)
            if (expected_class == predicted_class):
                correct_nb += 1
            #print("EXAMPLE #{}: expected={}, predicted={}{}".format(example_index, expected_class, predicted_class, " (INCORRECT)" if expected_class != predicted_class else ""))
            #self.draw.network_update(self)
        #print("SCORE : {}/{} ({}%)".format(correct_nb, VALIDATING[1] - VALIDATING[0], '%.2f' % (100 * correct_nb / (VALIDATING[1] - VALIDATING[0]))))
        for layer in self.layers[1:]:
            total_validation_loss += f.regularizations[self.regularization]["func"](layer.weights, self.reg_lambda)
        return total_validation_loss / (VALIDATING[1] - VALIDATING[0]), correct_nb

import util
import functions as f
import draw

import random
from datetime import datetime
import numpy as np
from types import SimpleNamespace

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
            raise ValueError(f"Activation function \"{activation}\" doesn't exist (available: {list(f.activations.keys())})")
        self.activation = activation
        self.act_params = act_params

class Network:
    def __init__(self):
        self.layers = []
        self.hp = SimpleNamespace(batch_size = 32,
                                  learning_rate = 0.01,
                                  learning_rate_decay = 0.2,
                                  learning_rate_min = 0.00005,
                                  loss = "mean_squared_error",
                                  regularization = "none",
                                  reg_lambda = 1e-3,
                                  save_frequency = 10)
        self.run_start = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.draw = None
        
    def __str__(self):
        s = ""
        for layer_index, layer in enumerate(self.layers):
            s += str(len(layer.outputs)) + "-" + layer.activation
            if layer_index < len(self.layers) - 1:
                s += "|"
        return s

    def set_hyperparameters(self, hp: SimpleNamespace):
        for key in vars(hp):
            if hasattr(self.hp, key):
                setattr(self.hp, key, getattr(hp, key))
                
        if self.hp.loss not in f.losses:
            raise ValueError(f"Loss function \"{self.hp.loss}\" doesn't exist (available: {list(f.losses.keys())})")
        if self.hp.regularization not in f.regularizations:
            raise ValueError(f"Regularization function \"{self.hp.regularization}\" doesn't exist (available: {list(f.regularizations.keys())})")
        for key in ['batch_size', 'learning_rate', 'reg_lambda', 'save_frequency']:
            value = getattr(self.hp, key, None)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Hyperparameter {key} \"{value}\" should be a positive number")
    
    def add_layer(self, size, activation, act_params=[]):
        self.layers.append(Layer(size, activation, act_params))

    def initialize_weights(self):
        if not "deri" in f.activations[self.layers[-1].activation]:
            raise ValueError(f"Activation function \"{self.layers[-1].activation}\" not suitable for output layer (no derivative implemented)")
        
        for layer_index, layer in enumerate(self.layers[1:], start=1):
            n_in = len(self.layers[layer_index - 1].outputs)
            if layer_index < len(self.layers) - 1:
                n_out = len(self.layers[layer_index + 1].outputs)
                if f.activations[layer.activation]["relu_family"]:
                    layer.weights = f.initializations["he_uniform"](n_in, n_out, len(layer.outputs))
                else:
                    layer.weights = f.initializations["xavier_uniform"](n_in, n_out, len(layer.outputs))
            else:
                layer.weights = f.initializations["output_layer"](n_in, n_out, len(layer.outputs))

    def feed_forward(self, example):
        self.layers[0].outputs = example.flatten() / 255 # make more generic
        for layer_index, layer in enumerate(self.layers[1:], start=1):
            inputs = self.layers[layer_index - 1].outputs
            layer.outputs = f.activations[layer.activation]["func"](np.dot(layer.weights, inputs) + layer.biases, layer.act_params)
    
    def backpropagate_error(self, expected_outputs):
        output_layer = self.layers[-1]
        if output_layer.activation == "softmax" and self.hp.loss == "cross_entropy":
            output_layer.errors = output_layer.outputs - expected_outputs
        else:
            loss_derivative = f.losses[self.hp.loss]["deri"](output_layer.outputs, expected_outputs)
            activation_derivative = f.activations[output_layer.activation]["deri"](output_layer.outputs)
            if output_layer.activation == "softmax":
                output_layer.errors = np.dot(loss_derivative, activation_derivative)
            else:
                output_layer.errors = loss_derivative * activation_derivative

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].errors = np.dot(self.layers[i + 1].weights.T, self.layers[i + 1].errors)

    def train_batch(self, examples, labels, batch_start, batch_end):
        if (batch_start == batch_end):
            return 0

        batch_gradients = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
        batch_biases = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
        batch_loss = 0
        for example_index in range(batch_start, batch_end):
            
            # Feed forward and compute loss
            self.feed_forward(examples[example_index])
            expected_outputs = one_hot(len(self.layers[-1].outputs), labels[example_index])
            batch_loss += f.losses[self.hp.loss]["func"](self.layers[-1].outputs, expected_outputs)

            # Backpropagate error and compute gradients
            self.backpropagate_error(expected_outputs)
            for layer_index, layer in enumerate(self.layers[1:], start=1):
                inputs = np.atleast_2d(self.layers[layer_index - 1].outputs)
                batch_gradients[layer_index - 1] += np.dot(np.atleast_2d(layer.errors).T, inputs)
                batch_biases[layer_index - 1] += layer.errors

        # Apply gradients to weights
        for layer_index, layer in enumerate(self.layers[1:], start=1):
            batch_loss += f.regularizations[self.hp.regularization]["func"](layer.weights, self.hp.reg_lambda)
            batch_gradients[layer_index - 1] += f.regularizations[self.hp.regularization]["deri"](layer.weights, self.hp.reg_lambda)
            #layer.weights -= np.clip((self.hp.learning_rate * batch_gradients[layer_index - 1]) / (batch_end - batch_start), -1, 1)
            layer.weights -= self.hp.learning_rate * batch_gradients[layer_index - 1] / (batch_end - batch_start)        
            layer.biases -= self.hp.learning_rate * batch_biases[layer_index - 1] / (batch_end - batch_start)

        return batch_loss
    
    def train(self, examples, labels, TRAINING, VALIDATING, epochs, hp):
        training_size, validating_size = TRAINING[1] - TRAINING[0], VALIDATING[1] - VALIDATING[0]
        self.set_hyperparameters(hp)
        self.draw.loss_init(self)
        print(
            f"TRAINING: examples={training_size}, batch_size={self.hp.batch_size}, epochs={epochs}, learning_rate={self.hp.learning_rate}, "
            f"loss={self.hp.loss}{', regularization=' + self.hp.regularization + ', reg_lambda=' + str(self.hp.reg_lambda) if self.hp.regularization != 'none' else ''}"
        )
    
        training_start = datetime.now()
        for epoch in range(1, epochs + 1):

            # Mini-batch training and validation
            training_loss = 0
            for batch_index in range(training_size // self.hp.batch_size):
                training_loss += self.train_batch(examples, labels, batch_index * self.hp.batch_size, (batch_index + 1) * self.hp.batch_size)
            training_loss += self.train_batch(examples, labels, (batch_index + 1) * self.hp.batch_size, (batch_index + 1) * self.hp.batch_size + training_size % self.hp.batch_size)
            training_loss /= training_size

            validation_loss, correct_nb = self.validate(examples, labels, VALIDATING)
            correct_rate = 100 * correct_nb / validating_size
            print(f"epoch {epoch}: t_loss={training_loss:.6f}, v_loss={validation_loss:.6f}, validation={correct_rate:.2f}% ({correct_nb}/{validating_size})")

            # Visualizations
            self.draw.t_loss_history.append(training_loss)
            self.draw.v_loss_history.append(validation_loss)
            self.draw.val_history.append(correct_rate)
            self.draw.network_update(self)
            self.draw.loss_update(self)

            # Learning rate decay and early stopping
            if (len(self.draw.v_loss_history) > 1 and self.draw.v_loss_history[-1] > self.draw.v_loss_history[-2]):
                if self.hp.learning_rate > self.hp.learning_rate_min + 1e-12:
                    self.hp.learning_rate = max(self.hp.learning_rate * self.hp.learning_rate_decay, self.hp.learning_rate_min)
                    print(f"decreasing learning rate to {self.hp.learning_rate:.9f}")
                    continue
                else:
                    print("early stopping (validation loss reached a minimum)")
                    break
            
            # Saving weights into a file
            if epoch % self.hp.save_frequency == 0:
                util.save_weights(self)
                
        util.save_weights(self)
        training_duration = datetime.now() - training_start
        print(f"TRAINING finished in {training_duration} ({training_duration / epoch} per epoch)")

    def validate(self, examples, labels, VALIDATING):
        total_validation_loss, correct_nb = 0, 0
        for example_index in range(VALIDATING[0], VALIDATING[1]):

            # Feed forward and compute loss
            self.feed_forward(examples[example_index])
            expected_class = labels[example_index]
            expected_outputs = one_hot(len(self.layers[-1].outputs), expected_class)
            total_validation_loss += f.losses[self.hp.loss]["func"](self.layers[-1].outputs, expected_outputs)

            # Check accuracy of network prediction
            predicted_class = np.argmax(self.layers[-1].outputs)
            if (expected_class == predicted_class):
                correct_nb += 1
            #print(f"EXAMPLE #{example_index}: expected={expected_class}, predicted={predicted_class}{' (INCORRECT)' if expected_class != predicted_class else ''}")
            #self.draw.network_update(self)
                
        for layer in self.layers[1:]:
            total_validation_loss += f.regularizations[self.hp.regularization]["func"](layer.weights, self.hp.reg_lambda)

        validating_size = VALIDATING[1] - VALIDATING[0]
        #print(f"SCORE : {correct_nb}/{validating_size} ({100 * correct_nb / validating_size:.2f}%)")
        return total_validation_loss / validating_size, correct_nb

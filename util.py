from pathlib import Path
from array import array
import struct
import numpy as np
import json
import os


def read_images(path):
    data_path = Path(path)
    with open(data_path, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = array("B", f.read())   
    images = []
    for i in range(size):
        image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images.append(image)
    return np.array(images)

def read_labels(path):
    data_path = Path(path)
    with open(data_path, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.array(array("B", f.read()))
    return labels


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_weights(net):
    network_weights = []
    network_weights.insert(0, {})
    network_weights[0]["metadata"] = net.__str__()
    network_weights[0]["loss"] = net.loss
    network_weights[0]["regularization"] = net.regularization
    network_weights[0]["reg_lambda"] = net.reg_lambda
    for layer_index, layer in enumerate(net.layers[1:], start=1):
        network_weights.insert(layer_index, {})
        network_weights[layer_index]["weights"] = layer.weights
        network_weights[layer_index]["biases"] = layer.biases

    save_dir = "weights"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = os.path.join(save_dir, "weights_" + net.run_start + ".json")
    save_file = open(save_filename, "w")
    json.dump(network_weights, save_file, cls=NumpyArrayEncoder)

def load_weights(net, weight_filename):
    save_filename = os.path.join("weights", weight_filename + ".json")
    save_file = open(save_filename, "r")
                                 
    network_weights = np.asarray(json.load(save_file))
    print("LOADING WEIGHTS - " + save_filename)
    print("Loading: " + network_weights[0]["metadata"])
    net.loss = network_weights[0]["loss"]
    net.regularization = network_weights[0]["regularization"]
    net.reg_lambda = network_weights[0]["reg_lambda"]
    for layer_index, layer in enumerate(network_weights[1:], start=1):
        if (len(layer["biases"]) != len(net.layers[layer_index].biases)):
            raise ValueError("Mismatch in size of layer {} between save file ({}) and network ({})".format(layer_index, len(layer["biases"]), len(net.layers[layer_index].biases)))
        net.layers[layer_index].weights = np.asarray(layer["weights"])
        net.layers[layer_index].biases = np.asarray(layer["biases"])

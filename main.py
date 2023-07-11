import util
import draw
import network

from types import SimpleNamespace

def main():
    net = network.Network()
    net.add_layer(784, activation="input")
    net.add_layer(128, activation="leakyrelu")
    net.add_layer(64, activation="leakyrelu")
    net.add_layer(32, activation="leakyrelu")
    net.add_layer(10, activation="softmax")
    print("NETWORK:", net)
    
    net.initialize_weights()
    #util.load_weights(net, "weights_20230710-151451")

    net.draw = draw.Draw(max_neurons_in_layer=12, output_labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    net.draw.network_init(net)
    
    images = util.read_images("mnist/train-images.idx3-ubyte")
    labels = util.read_labels("mnist/train-labels.idx1-ubyte")
    net.train(images, labels, TRAINING=(0, 50000), VALIDATING=(50000, 60000), epochs=500,
              hp=SimpleNamespace(learning_rate=0.01, loss="cross_entropy", regularization="none"))
    
    VALIDATING = (50000, 60000)
    validating_size = VALIDATING[1] - VALIDATING[0]
    _, correct_nb = net.validate(images, labels, VALIDATING)
    correct_rate = 100 * correct_nb / validating_size
    print(f"VALIDATION: {correct_rate:.2f}% ({correct_nb}/{validating_size})")

          
if __name__ == "__main__":
    main()

# neuralnetworks

This is a little neural network framework I started implementing from scratch in Python as a way to formally learn how they work after being fascinated by this topic as a hobbyist for a while. The code is rather simple to understand and probably not very optimized. Instead of hardcoding a single architecture to solve a task (the classic MNIST dataset), I tried to make it configurable in a lot of ways so I could experiment and adapt my network, and maybe allow me to reuse it for other tasks in the future.

The **main()** shows how it can be used to define a network. Layers can be added with a chosen activation function. You can then train the network for however long you want, and then do a validation test. It also has visualizations of the network, as well as the loss during training.

Here are some features and characteristics:
* Configurable number and size of layers
* System for saving weights into a JSON file during training, and loading these files back into the weights to skip training
* Strategy for weight initialization using He and Xavier methods (see **network.py/initialize_weights()** and **functions.py/initializations**)
* Several activation functions available (see **functions.py/activations**)
* Several loss functions available (see **functions.py/losses**)
* Shortcut for softmax/cross-entropy combination
* Vectorization of most operations
* Mini-batch gradient descent
* Several regularization methods available (see **functions.py/regularizations**)
* Early stopping based on monitoring of the validation loss during training

Things that I may or may not look into (issues, features or further optimization techniques):
* dropout/pruning
* learning rate scheduling/warmup
* batch normalization
* investigate gaussian (exploding gradients?)
* display pictures
* improve loss function visualisation
* fix janky draw_network
* command line options
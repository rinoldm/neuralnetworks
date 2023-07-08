import network
import functions as f

import matplotlib.pyplot as plt
import numpy as np
import time

class Draw:
    def __init__(self, max_neurons_in_layer, output_labels=[]):
        self.MAX_NEURONS_IN_LAYER = max_neurons_in_layer
        self.HALF_SHORT_LAYER = int(self.MAX_NEURONS_IN_LAYER / 2) - 1
        self.output_labels = output_labels
        self.circles = []
        self.lines = []
        self.texts = {}
        self.network_fig = None

        self.loss_fig = None
        self.loss_ax = None
        self.t_loss_history = []
        self.t_loss_line = None
        self.v_loss_history = []
        self.v_loss_line = None
        self.val_ax = None
        self.val_history = []
        self.val_line = None
        

    def network_init(self, net):
        self.network_fig = plt.figure(figsize=(6, 6), num="Network state")
        self.network_fig.patch.set_facecolor('#b3e6ff')
        ax = self.network_fig.gca()
        ax.axis('off')
        plt.ylim(-0.5, 0.5)
        plt.ion()

        largest_layer_size = min(self.MAX_NEURONS_IN_LAYER, max(map(lambda x: len(x.outputs), net.layers)))
        circle_xspacing = 1 / (len(net.layers) + 1)
        circle_yspacing = 1 / (largest_layer_size + 1)
        for layer_index, layer in enumerate(net.layers):
            self.circles.insert(layer_index, [])
            self.lines.insert(layer_index, [])
            self.texts["output_labels"] = []
            circle_x = circle_xspacing * (layer_index + 1)
            plt.text(circle_x, -0.52, "Input" if layer_index == 0 else "Output" if layer_index == len(net.layers) - 1 else "Hidden", ha='center', va='center')
            plt.text(circle_x, -0.55, len(layer.outputs), ha='center', va='center')
            plt.text(circle_x, -0.58, layer.activation if layer_index > 0 else "-", ha='center', va='center')
            layer_top = circle_yspacing * (min(len(layer.outputs), self.MAX_NEURONS_IN_LAYER) - 1) / 2
            displayed_outputs = layer.outputs if len(layer.outputs) <= self.MAX_NEURONS_IN_LAYER else np.concatenate((layer.outputs[:self.HALF_SHORT_LAYER], layer.outputs[-self.HALF_SHORT_LAYER:]))
            for neuron_index, output in enumerate(displayed_outputs):
                circle_y = layer_top - circle_yspacing * (neuron_index + (2 if len(layer.outputs) > self.MAX_NEURONS_IN_LAYER and neuron_index >= self.HALF_SHORT_LAYER else 0))
                circle = plt.Circle((circle_x, circle_y), radius=circle_yspacing / 3,
                                    facecolor=str(output), edgecolor='black', zorder=10)
                self.circles[layer_index].insert(neuron_index, circle)
                self.lines[layer_index].insert(neuron_index, [])
                ax.add_artist(circle)
                if layer_index > 0:
                    displayed_weights = layer.weights[neuron_index] if len(net.layers[layer_index - 1].outputs) <= self.MAX_NEURONS_IN_LAYER else np.concatenate((layer.weights[neuron_index][:self.HALF_SHORT_LAYER], layer.weights[neuron_index][-self.HALF_SHORT_LAYER:]))
                    for weight_index, weight in enumerate(displayed_weights):
                        line = plt.Line2D([circle_x, self.circles[layer_index - 1][weight_index].center[0]],
                                          [circle_y, self.circles[layer_index - 1][weight_index].center[1]])
                        self.lines[layer_index][neuron_index].insert(weight_index, line)
                        ax.add_artist(line)
                if layer_index == len(net.layers) - 1:
                    text = plt.text(circle_x + circle_yspacing / 3 + 0.01, circle_y, '%.3f' % output, va='center')
                    self.texts["output_labels"].insert(neuron_index, text)
            if (len(layer.outputs) > self.MAX_NEURONS_IN_LAYER):
                for etc_circle_index in range(3):
                    etc_circle = plt.Circle((circle_x, -circle_yspacing / 2 + etc_circle_index * circle_yspacing / 2),
                                            radius=circle_yspacing / 8, color='black', zorder=10)
                    ax.add_artist(etc_circle)
     
        self.network_update(net)
        plt.show()
        
    def network_update(self, net):
        for layer_index, layer in enumerate(net.layers):
            displayed_outputs = layer.outputs if len(layer.outputs) <= self.MAX_NEURONS_IN_LAYER else np.concatenate((layer.outputs[:self.HALF_SHORT_LAYER], layer.outputs[-self.HALF_SHORT_LAYER:]))
            for neuron_index, output in enumerate(displayed_outputs):
                greyscale_value = f.activations[layer.activation]["norm"](output)
                self.circles[layer_index][neuron_index].set_facecolor(str(greyscale_value))
                if layer_index > 0:
                    displayed_weights = layer.weights[neuron_index] if len(net.layers[layer_index - 1].outputs) <= self.MAX_NEURONS_IN_LAYER else np.concatenate((layer.weights[neuron_index][:self.HALF_SHORT_LAYER], layer.weights[neuron_index][-self.HALF_SHORT_LAYER:]))
                    for weight_index, weight in enumerate(displayed_weights):
                        self.lines[layer_index][neuron_index][weight_index].set_color('red' if weight < 0 else 'green')
                        self.lines[layer_index][neuron_index][weight_index].set_linewidth(2 * abs(weight) / abs(max(displayed_weights, key=abs)))
                if layer_index == len(net.layers) - 1:
                    self.texts["output_labels"][neuron_index].set_text('%.6f' % output + ((" → " + self.output_labels[neuron_index]) if len(self.output_labels) > 0 else ""))
        self.network_fig.canvas.flush_events()

    def loss_init(self, net):
        plt.ion()
        self.loss_fig, self.loss_ax = plt.subplots()
        self.loss_fig.canvas.manager.set_window_title('Loss function')
        
        self.loss_ax.set_yscale("log")
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.t_loss_line, = self.loss_ax.plot([], [], color='tab:purple')
        self.v_loss_line, = self.loss_ax.plot([], [], color='tab:red')

        self.val_ax = self.loss_ax.twinx()
        self.val_ax.set_ylabel('Validation')
        self.val_ax.set_ylim([0, 100])
        self.val_line, = self.val_ax.plot([], [], color='tab:green')

    def loss_update(self, net):
        self.t_loss_line.set_data(list(range(len(self.t_loss_history))), self.t_loss_history)
        self.v_loss_line.set_data(list(range(len(self.v_loss_history))), self.v_loss_history)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view(True, True, True)

        self.val_line.set_data(list(range(len(self.val_history))), self.val_history)
        
        self.loss_fig.canvas.draw()
        self.loss_fig.canvas.flush_events()
        

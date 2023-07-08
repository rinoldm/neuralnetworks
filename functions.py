import numpy as np
import scipy.special

def clipexp(x):
    return np.exp(np.clip(x, -50, 50))

def init_w(n_in, size, func):
    return np.array([[func() for _ in range(n_in)] for _ in range(size)])


initializations = {
    "he_normal": lambda n_in, n_out, size : init_w(n_in, size, lambda : np.random.normal(0, np.sqrt(2 / n_in))),
    "he_uniform": lambda n_in, n_out, size : init_w(n_in, size, lambda : np.random.uniform(-np.sqrt(6 / n_in), np.sqrt(6 / n_in))),
    "xavier_normal": lambda n_in, n_out, size : init_w(n_in, size, lambda : np.random.normal(0, np.sqrt(2 / (n_in + n_out)))),
    "xavier_uniform": lambda n_in, n_out, size : init_w(n_in, size, lambda : np.random.uniform(-np.sqrt(6 / (n_in + n_out)), np.sqrt(6 / (n_in + n_out)))),
    "output": lambda n_in, n_out, size : init_w(n_in, size, lambda : np.random.normal(0, 0.1)),
}

norms = {
    "a_to_b_lin": lambda x, a, b: (x - a) / (b - a),
    "0_to_1": lambda x : norms["a_to_b_lin"](x, 0, 1),
    "-1_to_1": lambda x : norms["a_to_b_lin"](x, -1, 1),
    "0_to_inf": lambda x : 0 if x <= 0 else (1 - 1 / (x + 1)),
}

activations = {
    "input": {
        "norm": norms["0_to_1"],
    },
    "identity": {
        "func": lambda x, p : x,
        "norm": norms["0_to_inf"],
        "relu_family": False,
    },
    "step": {
        "func": lambda x, p : np.where(x < 0, 0, 1),
        "norm": norms["0_to_1"],
        "relu_family": False,
    },
    "softmax": {
        "func": lambda x, p : scipy.special.softmax(x - np.max(x)),
        "deri": lambda out : [[out[i] * (1 - out[j]) if i == j else -out[i] * out[j] for j in range(len(out))] for i in range(len(out))],
        "norm": norms["0_to_1"],
        "relu_family": False,
    },
    "sigmoid": {
        "func": lambda x, p : 1 / (1 + clipexp(-x)),
        "deri": lambda y : y * (1 - y),
        "norm": norms["0_to_1"],
        "relu_family": False,
    },
    "tanh": {
        "func": lambda x, p : np.tanh(x),
        "norm": norms["-1_to_1"],
        "relu_family": False,
    },
    "prelu": {
        "func": lambda x, p : np.where(x < 0, p[0] * x, x),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "relu": {
        "func": lambda x, p : activations["prelu"]["func"](x, [0]),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "leakyrelu": {
        "func": lambda x, p : activations["prelu"]["func"](x, [0.01]),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "gelu": {
        "func": lambda x, p : x * (1 + scipy.special.erf(x / np.sqrt(2))) / 2,
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "softplus": {
        "func": lambda x, p : np.log(1 + clipexp(x)),
        "norm": norms["0_to_inf"],
        "relu_family": False,
    },
    "selu": {
        "func": lambda x, p : np.where(x < 0, p[1] * (p[0] * clipexp(x - 1)), x),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "selu_orig": {
        "func": lambda x, p : activations["selu"]["func"](x, [1.67326, 1.0507]),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "elu": {
        "func": lambda x, p : activations["selu"]["func"](x, [p[0], 1]),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "silu": {
        "func": lambda x, p : x / (1 + clipexp(-x)),
        "norm": norms["0_to_inf"],
        "relu_family": True,
    },
    "gaussian": {
        "func": lambda x, p : clipexp(-x**2),
        "norm": norms["0_to_1"],
        "relu_family": False,
    },
}

losses = {
    "mean_squared_error": {
        "func": lambda pred, exp : np.average((pred - exp)**2) / 2,
        "deri": lambda pred, exp : pred - exp,
    },
    "cross_entropy": {
        "func": lambda pred, exp, epsilon=1e-12 : -np.sum(exp * np.log(np.clip(pred, epsilon, 1. - epsilon) + epsilon)) / len(pred),
        "deri": lambda pred, exp, epsilon=1e-12 : -exp / (pred + epsilon) / len(pred),
    },
    #binary cross_entropy
}

regularizations = {
    "none": {
        "func": lambda w, reg_lambda : 0,
        "deri": lambda w, reg_lambda : 0,
    },
    "L1": {
        "func": lambda w, reg_lambda : reg_lambda * np.sum(np.abs(w)),
        "deri": lambda w, reg_lambda : np.where(w > 0, reg_lambda, -reg_lambda),
    },
    "L2": {
        "func": lambda w, reg_lambda : reg_lambda * np.sum(np.square(w)),
        "deri": lambda w, reg_lambda : 2 * reg_lambda * w,
    },
}

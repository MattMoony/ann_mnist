#!/usr/bin/env python3

import PIL
import gzip
import numpy as np
import matplotlib.pyplot as plt
from typing import *

plt.ioff()

def read_features(fname: str, num: int, img_size: int = 28) -> np.matrix:
    """Read `num` images from the dataset and return them as an `np.matrix` of size (num, img_size * img_size)"""
    with gzip.open(fname) as f:
        f.read(16)
        data: np.ndarray = np.frombuffer(f.read(img_size * img_size * num), dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, img_size * img_size)
    return np.asmatrix(data)

def read_labels(fname: str, num: int, label_size: int = 1) -> np.matrix:
    """Read `num` labels from the dataset and return them as an `np.matrix` of size (num, label_size)"""
    with gzip.open(fname) as f:
        f.read(8)
        labels: np.ndarray = np.frombuffer(f.read(num * label_size), dtype=np.uint8).astype(np.uint8)
        labels = labels.reshape(num, label_size)
    return np.asmatrix(labels)

def plot_pixel_range(img: np.ndarray) -> None:
    """Plot the pixel value distribution + the image using matplotlib"""
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.violinplot(img.flatten())
    plt.title('Pixel value distribution')
    plt.xlabel('Probability density')
    plt.ylabel('Pixel value')
    plt.subplot(122)
    plt.imshow(img.reshape(28, -1), cmap='gray')
    plt.show()

def normalize_features(x: np.matrix) -> np.matrix:
    """Normalizes the features to be fed into the neural network"""
    mu: np.ndarray = np.mean(x, axis=0)
    sigma: np.ndarray = np.max(x, axis=0) - np.min(x, axis=0)
    sigma[sigma < 1] = 1
    return (x - mu) / sigma

def loss(p: np.matrix, y: np.matrix, lam: float = 0, weights: List[np.matrix] = []) -> float:
    """Compute the (regularized) CCE for the given predictions + labels"""
    return (-1/y.shape[0]) * np.sum(np.log(np.take_along_axis(p.T, y, axis=1))) + (lam/(2 * y.shape[0])) * sum(np.sum(np.power(w, 2)) for w in weights)

def softmax(z: np.matrix) -> np.matrix:
    """Applies the softmax activation function to the given values"""
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def loss_g(p: np.matrix, y: np.matrix) -> np.matrix:
    """Compute the neurons' gradients with regard to the loss (softmax has to be applied before loss computation)"""
    grad: np.matrix = softmax(p)
    np.put_along_axis(grad.T, y, np.take_along_axis(grad.T, y, axis=1) - 1, axis=1)     # subtract 1 from the "label" neurons
    return (1/y.shape[0]) * grad                                                        # return the gradients normalized by the batch size

def sigmoid(z: np.matrix) -> np.matrix:
    """Applies the logistic sigmoid function to the given values"""
    return 1 / (1 + np.exp(-z))

def tanh(z: np.matrix) -> np.matrix:
    """Applies the TanH activation function to the given values"""
    return np.tanh(z)

def relu(z: np.matrix) -> np.matrix:
    """Applies the ReLU activation function to the given values"""
    return z.clip(0)

def plot_actf(f: Union[Callable[[np.matrix], np.matrix], List[Callable[[np.matrix], np.matrix]]], low: int = -10, high: int = 10, n: int = 200):
    """Run some values through some activation functions and plot the result"""
    plt.figure(figsize=(10,4))
    if type(f) != list:
        f = [f,]
    for i, x in enumerate(f):
        plt.subplot(1, len(f), i+1)
        plt.title(x.__name__)
        plt.plot(np.linspace(low, high, n), np.asarray(x(np.asmatrix(np.linspace(low, high, n)))).flatten())
    plt.show()

def sigmoid_g(z: np.matrix) -> np.matrix:
    """Computes the gradients of the sigmoid function"""
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def tanh_g(z: np.matrix) -> np.matrix:
    """Computes the gradients of the TanH function"""
    return 1 - np.power(tanh(z), 2)

def relu_g(z: np.matrix) -> np.matrix:
    """Computes the gradients of the ReLU function"""
    r: np.matrix = np.zeros(z.shape)
    r[z>0] = 1
    return r

def step(x: np.matrix, y: np.matrix, weights: List[np.matrix], actf: List[Callable[[np.matrix], np.matrix]],
         actf_g: List[Callable[[np.matrix], np.matrix]], lam: float = .03) -> List[np.matrix]:
    """
    Take a single training step - i.e. complete a training epoch and return the computed gradients
    :param x: A mini-batch of features
    :param y: The labels for the mini-batch
    :param weights: The weights of the network
    :param actf: The activiation functions of the networks' layers
    :param actf_g: The derivations of the activation functions
    :param lam: The regularization parameter lambda - for weight regularization
    :return: The weights' computed gradients
    """

    assert actf[-1] == softmax          # last activation function has to be softmax
    assert len(weights) == len(actf)    # every layer has to have an activation function
    assert len(actf) == len(actf_g) + 1 # every activation fun. must have a derivation (except last, since part of loss derivative)
    assert x.shape[1] == y.shape[0]     # every batch entry has to have a label

    l_val: List[np.matrix] = [ x, ]     # the raw values of a network layer
    l_act: List[np.matrix] = [ x, ]     # the activations of a network layer

    # FORWARD PROPAGATION

    for w, a in zip(weights, actf):
        l_val.append(w.T * np.vstack(( l_act[-1], np.ones(l_act[-1].shape[1]) )))       # add all ones to act as bias nodes
        l_act.append(a(l_val[-1]))                                                      # pass values through activation function
    
    # BACKPROPAGATION

    l_grad: List[np.matrix] = [ loss_g(l_val[-1], y), ]
    w_grad: List[np.matrix] = []

    for i, (w, d) in enumerate(reversed(list(zip(weights, [ *actf_g, lambda x: np.ones(x.shape), ]))), 1):
        delta: np.matrix = np.multiply(l_grad[-1], d(l_val[-i]))
        w_grad.append(np.vstack(( l_val[-i-1], np.ones(l_val[-i-1].shape[1]) )) * delta.T + (lam/y.shape[0]) * w)
        l_grad.append(w[:-1,:] * delta)

    return list(reversed(w_grad))

def predict(x: np.matrix, weights: List[np.matrix], actf: List[Callable[[np.matrix], np.matrix]]) -> np.matrix:
    """
    Predict the classes for the given input - i.e. run the input through the network
    :param x: The input features
    :param weights: The weights of the network
    :param actf: The activiation functions of the networks' layers
    :return: The network's output - i.e. its prediction
    """
    activation: np.matrix = x
    for w, a in zip(weights, actf):
        activation = a(w.T * np.vstack(( activation, np.ones(activation.shape[1]) )))
    return activation

def make_class(p: np.matrix) -> np.matrix:
    """Turns the raw output of the network into actual class numbers"""
    return np.argmax(p, axis=0)

def train(x: np.matrix, y: np.matrix, weights: List[np.matrix], actf: List[Callable[[np.matrix], np.matrix]],
          actf_g: List[Callable[[np.matrix], np.matrix]], optimizer: Callable[..., np.matrix], epochs: int, 
          optimizer_args: List[Any] = [], optimizer_kwargs: Dict[str, Any] = {}, batch_size: int = 32, 
          lam: float = .03, all_loss: bool = False) -> Union[List[np.matrix], Tuple[List[np.matrix], List[float], List[float]]]:
    """
    Perform Mini-Batch Gradient Descent on the given dataset using the given weights & parameters
    :param x: The feature part of the dataset
    :param y: The labels of the dataset
    :param weights: The weights of the network
    :param actf: The activiation functions of the networks' layers
    :param actf_g: The derivations of the activation functions
    :param optimizer: The optimizer to use for weight adjustion
    :param epochs: The number of epochs to train for
    :param optimizer_args: The positional arguments for the optimizer function
    :param optimizer_kwargs: The keyword arguments for the optimizer function
    :param batch_size: The batch-size to use for mini-batch gradient descent
    :param lam: The regularization parameter lambda - for weight regularization
    :param all_loss: Compute + return a list of losses for every epoch
    :return: The newly trained weights (+ optionally loss per epoch)
    """
    weights: List[np.matrix] = [ w.copy() for w in weights ]
    losses: List[float] = []
    accuracies: List[float] = []

    for i in range(epochs):
        if i%(epochs//10) == 0:
            print(f'[{i:04d}/{epochs:04d}] Loss = {loss(predict(x.T, weights, actf), y)} ... ')
        if all_loss and i%(epochs//100) == 0:
            losses.append(loss(predict(x.T, weights, actf), y))
            accuracies.append(accuracy(make_class(predict(x.T, weights, actf)).T, y))
        choices: np.ndarray = np.random.choice(np.arange(x.shape[0]), batch_size)
        grad: List[np.matrix] = step(x[choices].T, y[choices], weights, actf, actf_g, lam)
        for j in range(len(weights)):
            weights[j] = optimizer(weights[j], grad[j], j, *optimizer_args, **optimizer_kwargs)
    return (weights, losses, accuracies) if all_loss else weights

def vanilla(weight: np.matrix, grad: np.matrix, i: int, alpha: float) -> np.matrix:
    """
    Given a weight matrix and its gradient, perform vanilla gradient descent
    :param weight: The weight matrix connecting two layers in a neural net
    :param grad: The corresponding gradients
    :param i: The index of the weights that are being updated
    :param alpha: The learning rate "alpha"
    :return: The new weight matrix
    """
    return weight - alpha * grad

def make_momentum(neurons: List[int], gamma: float) -> Callable[[np.matrix, np.matrix, float], np.matrix]:
    """
    Wrapper function to create a new gradient descent w. momentum optimizer
    :param neurons: The number of neurons at each layer - the network's shape
    :param gamma: The hyperparameter for momentum decay
    :return: An instance of a momentum optimizer
    """
    v: List[np.matrix] = [ np.zeros((neurons[i]+1, neurons[i+1])) for i in range(len(neurons)-1) ]

    def momentum(weight: np.matrix, grad: np.matrix, i: int, alpha: float) -> np.matrix:
        """
        Given a weight matrix and its gradient, perform gradient descent w. momentum
        :param weight: The weight matrix connecting two layers in a neural net
        :param grad: The corresponding gradients
        :param i: The index of the weights that are being updated
        :param alpha: The learning rate "alpha"
        :return: The new weight matrix
        """ 
        v[i] = gamma * v[i] + alpha * grad
        return weight - v[i]
    
    return momentum

def make_adam(neurons: List[int], beta1: float, beta2: float, epsilon: float = 1e-7) -> Callable[[np.matrix, np.matrix, float], np.matrix]:
    """
    Wrapper function to create a new Adam optimizer
    :param neurons: The number of neurons at each layer - the network's shape
    :param beta1: The hyperparameter for first order moment decay
    :param beta2: The hyperparameter for second order moment decay
    :param epsilon: A hyperparameter to prevent divisions by zero
    :return: An instance of an Adam optimizer
    """
    m: List[np.matrix] = [ np.zeros((neurons[i]+1, neurons[i+1])) for i in range(len(neurons)-1) ]
    v: List[np.matrix] = [ np.zeros((neurons[i]+1, neurons[i+1])) for i in range(len(neurons)-1) ]

    def adam(weight: np.matrix, grad: np.matrix, i: int, alpha: float) -> np.matrix:
        """
        Given a weight matrix and its gradient, perform gradient descent w. AdaDelta
        :param weight: The weight matrix connecting two layers in a neural net
        :param grad: The corresponding gradients
        :param i: The index of the weights that are being updated
        :param alpha: The learning rate "alpha"
        :return: The new weight matrix
        """
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * np.power(grad, 2)
        m_s: np.matrix = m[i] / (1 - beta1)
        v_s: np.matrix = v[i] / (1 - beta2)
        return weight - alpha * np.multiply(m_s, 1 / np.sqrt(v_s + epsilon))

    return adam

def uniform_init(neurons: List[int]) -> List[np.matrix]:
    """
    Generates initalized weights for a network of the given size
    :param neurons: The number of neurons in each layer
    :return: The uniformly randomly initalized weights of the network
    """
    return [ np.asmatrix(np.random.uniform(-1, 1, (neurons[i]+1, neurons[i+1]))) for i in range(len(neurons)-1) ]

def standard_init(neurons: List[int]) -> List[np.matrix]:
    """
    Generates initalized weights for a network of the given size
    :param neurons: The number of neurons in each layer
    :return: The "Standard"-initalized weights of the network
    """
    return [ np.asmatrix(np.random.uniform(-1, 1, (neurons[i]+1, neurons[i+1])) * np.sqrt(1 / (neurons[i]+1))) for i in range(len(neurons)-1) ]

def xavier_init(neurons: List[int]) -> List[np.matrix]:
    """
    Generates initalized weights for a network of the given size
    :param neurons: The number of neurons in each layer
    :return: The "Xavier"-initalized weights of the network
    """
    return [ np.asmatrix(np.random.uniform(-1, 1, (neurons[i]+1, neurons[i+1])) * np.sqrt(6 / (neurons[i]+1 + neurons[i+1]))) for i in range(len(neurons)-1) ]

def he_init(neurons: List[int]) -> List[np.matrix]:
    """
    Generates initialized weights for a network of the given size
    :param neurons: The number of neurons in each layer
    :return: The "He"-initalized weights of the network
    """
    return [ np.asmatrix(np.vstack(( np.random.uniform(-1, 1, (neurons[i], neurons[i+1])) * np.sqrt(2 / neurons[i]), np.zeros((1, neurons[i+1])) ))) for i in range(len(neurons)-1) ]

def accuracy(p: np.matrix, y: np.matrix) -> float:
    """Calculates the accuracy of the given predictions [%]"""
    assert p.shape == y.shape
    return 1 - (np.count_nonzero(p - y) / y.shape[0])

def plot_pred(x: np.matrix, y: np.matrix, weights: List[np.matrix], actf: List[Callable[[np.matrix], np.matrix]], 
              size: Tuple[int, int], labels: bool = True, fname: Optional[str] = None) -> None:
    """Plot some predictions using the given network, features and labels"""
    plt.figure(figsize=(10, 5))
    for i in range(size[0]):
        for j in range(size[1]):
            plt.subplot(*size, i*size[1]+j+1)
            choice: int = np.random.randint(0, x.shape[0])
            r: np.matrix = x[choice]
            p: int = np.asarray(make_class(predict(r.T, weights, actf))).flatten()[0]
            l: int = np.asarray(y)[choice][0]
            plt.imshow(r.reshape(28, 28), cmap='summer' if p == l else 'spring')
            if labels:
                plt.title(f'{p} (actually {l})')
            else:
                plt.axis('off')
    if fname:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

def load_transparent(fname: str) -> PIL.Image:
    """
    Makes the background of an image transparent - for usage with `imageio`
    Thanks @FelixHo (https://stackoverflow.com/questions/46850318/transparent-background-in-gif-using-python-imageio#51219787)
    """
    im: PIL.Image.Image = PIL.Image.open(fname)
    alpha: PIL.Image.Image = im.getchannel('A')
    im = im.convert('RGB').convert('P', palette=PIL.Image.ADAPTIVE, colors=255)
    mask = PIL.Image.eval(alpha, lambda a: 255 if a <=128 else 0)
    im.paste(255, mask)
    im.info['transparency'] = 255
    return im

def plot_loss_acc(losses: List[List[float]], accuracies: List[List[float]], legend: List[str], 
                  epochs: int, start: int = 0) -> None:
    """
    Plot losses + accuracies in respect to epochs - useful
    for comparison between different network strategies
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss / Epoch')
    for l in losses:
        plt.plot(np.linspace(start, epochs, len(l)), l)
    plt.legend(legend)
    plt.subplot(122)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy / Epoch')
    for a in accuracies:
        plt.plot(np.linspace(start, epochs, len(a)), a)
    plt.legend(legend)
    plt.show()

def main():
    # DATASET PREPARATION

    X_TRAIN: np.matrix  = read_features('data/train-images-idx3-ubyte.gz', 60_000)
    Y_TRAIN: np.matrix  = read_labels('data/train-labels-idx1-ubyte.gz', 60_000)
    X_TEST: np.matrix   = read_features('data/t10k-images-idx3-ubyte.gz', 10_000)
    Y_TEST: np.matrix   = read_labels('data/t10k-labels-idx1-ubyte.gz', 10_000)

    # plot_pixel_range(np.asarray(X_TRAIN[0]))
    # plot_pixel_range(np.asarray(X_TEST[0]))

    X_TRAIN = normalize_features(X_TRAIN)
    X_TEST = normalize_features(X_TEST)

    # plot_pixel_range(np.asarray(X_TRAIN[0]))
    # plot_pixel_range(np.asarray(X_TEST[0]))

    # MODEL DEFINITION

    # plot_actf([sigmoid, tanh, relu,])
    
    neurons: List[int] = [ 28*28, 800, 10, ]        # specifies the number of neurons at each layer
    weights: List[np.matrix] = he_init(neurons)     # creates + initializes the weights
    actf: List[Callable[[np.matrix], np.matrix]] = [ relu, ] * (len(weights) - 1) + [ softmax, ]        # the activation functions
    actf_g: List[Callable[[np.matrix], np.matrix]] = [ relu_g, ] * (len(weights) - 1)                   # their derivations

    # TRAINING

    ALPHA: float        = .005      # the learning rate, α
    LAMBDA: float       = .03       # the regularization parameter to prevent weights from exploding, λ
    GAMMA: float        = .9        # a decay parameter for optimizers, γ
    BETA1: float        = .9        # a decay parameter for Adam, β_1
    BETA2: float        = .999      # a decay parameter for Adam, β_2
    EPSILON: float      = 1e-7      # a parameter to prevent divison by zero in multiple optimizers, ε
    BATCH_SIZE: int     = 64        # the batch size for multi-batch SGD
    EPOCHS: int         = 1024      # the number of epochs (= iterations) to train for

    n_weights: List[np.matrix] = train(X_TRAIN, Y_TRAIN, weights, actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, 
                                    optimizer_args=[ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA)

    # EVALUATION

    print(f'[*] Accuracy (training): {accuracy(make_class(predict(X_TRAIN.T, n_weights, actf)).T, Y_TRAIN)*100:.3f}% ... ')
    print(f'[*] Accuracy (testing): {accuracy(make_class(predict(X_TEST.T, n_weights, actf)).T, Y_TEST)*100:.3f}% ... ')

    plot_pred(X_TEST, Y_TEST, n_weights, actf, (5, 10), labels=False)

    # EXPERIMENTING

    # vanilla_weights, vanilla_loss, vanilla_accuracies = train(X_TRAIN, Y_TRAIN, weights, actf, actf_g, vanilla, EPOCHS, [.05,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # momentum_weights, momentum_loss, momentum_accuracies = train(X_TRAIN, Y_TRAIN, weights, actf, actf_g, make_momentum(neurons, GAMMA), EPOCHS, [.05,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # adam_weights, adam_loss, adam_accuracies = train(X_TRAIN, Y_TRAIN, weights, actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)

    # plot_loss_acc([ vanilla_loss, momentum_loss, adam_loss, ], 
    #             [ vanilla_accuracies, momentum_accuracies, adam_accuracies, ], 
    #             [ 'Vanilla MB-SGD', 'MB-SGD w. Momentum', 'Adam',], 
    #             EPOCHS)

    # ign_off: int = 3
    # plot_loss_acc([ momentum_loss[ign_off:], adam_loss[ign_off:], ],
    #             [ momentum_accuracies[ign_off:], adam_accuracies[ign_off:], ],
    #             [ 'MB-SGD w. Momentum', 'Adam', ],
    #             EPOCHS, start=(EPOCHS//100)*ign_off)

    # sigmoid_weights, sigmoid_loss, sigmoid_accuracies = train(X_TRAIN, Y_TRAIN, weights, [ sigmoid, ] * (len(weights)-1) + [ softmax, ], [ sigmoid_g, ] * (len(weights)-1), make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # tanh_weights, tanh_loss, tanh_accuracies = train(X_TRAIN, Y_TRAIN, weights, [ tanh, ] * (len(weights)-1) + [ softmax, ], [ tanh_g, ] * (len(weights)-1), make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # relu_weights, relu_loss, relu_accuracies = train(X_TRAIN, Y_TRAIN, weights, [ relu, ] * (len(weights)-1) + [ softmax, ], [ relu_g, ] * (len(weights)-1), make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)

    # plot_loss_acc([ sigmoid_loss, tanh_loss, relu_loss, ],
    #             [ sigmoid_accuracies, tanh_accuracies, relu_accuracies, ],
    #             [ 'Logistic Sigmoid', 'TanH', 'ReLU', ],
    #             EPOCHS)

    # uniform_weights, uniform_loss, uniform_accuracies = train(X_TRAIN, Y_TRAIN, uniform_init(neurons), actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # standard_weights, standard_loss, standard_accuracies = train(X_TRAIN, Y_TRAIN, standard_init(neurons), actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # xavier_weights, xavier_loss, xavier_accuracies = train(X_TRAIN, Y_TRAIN, xavier_init(neurons), actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)
    # he_weights, he_loss, he_accuracies = train(X_TRAIN, Y_TRAIN, he_init(neurons), actf, actf_g, make_adam(neurons, BETA1, BETA2, EPSILON), EPOCHS, [ALPHA,], batch_size=BATCH_SIZE, lam=LAMBDA, all_loss=True)

    # plot_loss_acc([ uniform_loss, standard_loss, xavier_loss, he_loss, ],
    #             [ uniform_accuracies, standard_accuracies, xavier_accuracies, he_accuracies, ],
    #             [ 'Uniform Init', 'Standard Init', 'Xavier Init', 'He Init', ],
    #             EPOCHS)

    # ign_off: int = 3
    # plot_loss_acc([ standard_loss[ign_off:], xavier_loss[ign_off:], he_loss[ign_off:], ],
    #             [ standard_accuracies[ign_off:], xavier_accuracies[ign_off:], he_accuracies[ign_off:], ],
    #             [ 'Standard Init', 'Xavier Init', 'He Init', ],
    #             EPOCHS, start=ign_off*(EPOCHS//100))

if __name__ == '__main__':
    main()

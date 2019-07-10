import random
import numpy as np

class Network(object):
	
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        # typically, use y to represent layer, x to represent node in layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """return output of network when input layer is set to a (array)"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k 
                            in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # update the network biases and weights based on this batch
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights & biases using gradient descent with 
           backpropagation to a single mini_batch.
           mini batch is a list of ''(x, y)'' tuples, and ''eta'' is the 
           learning rate
        """
        # preallocate arrays for nabla_b and nabla_w
        x, y = zip(*mini_batch)
        x = np.hstack(x)
        y = np.hstack(y)
        nabla_b, nabla_w = self.backprop(x, y)
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in 
                            zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in 
                            zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """Return a tuple ''(nabla_b, nabla_w)'' representing the gradient for
           the cost function C_x. (ie difference between expected (y) and 
           actual result given x).
           ''nabla_b'' and ''nabla_w'' are layer_by_layer lists of numpy arrays,
           similar to ''self.biases'' and ''self.weights''.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z) 
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w
    
    def evaluate(self, test_data):
        """Returns number of inputes for which the correct result is found. The 
           result is given as the neuron in the final layer with the highest
           activation
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Returns the vector of partial derivates '' dell C_x / dell a '' for
           the output activations.
        """
        return output_activations - y
    
    
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """derivative of the sigmoid prime function"""
    return sigmoid(z) * (1 - sigmoid(z))

            
            
            
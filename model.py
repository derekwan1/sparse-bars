import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, lamb):
    return 1 / (1 + np.exp(-x))

class Model:
    def __init__(self, params):
        """
        Generate `neurons` of size (num_neurons x dims)
        I'll use the X*W convention where W is my weight
        matrix and X is the input data

        Also generate `inhib` which are Hebbian inhibitory
        connections. `inhib[i, j]` should be the weight between
        neurons i and j

        `thresholds` are our perceptron thresholds,
        which I keep separate from neurons because they follow
        different update rules 
        """
        num_neurons = params["num_neurons"]
        input_dims = params["input_dims"]

        self.neurons = np.random.randn(input_dims, num_neurons)
        self.input_dims = input_dims
        self.inhib = np.zeros((num_neurons, num_neurons))
        self.thresholds = np.reshape(np.ones((num_neurons)), (num_neurons, 1))
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.lamb = params["lamb"]
        self.epsilon = params["epsilon"]
        self.p = params["p"]
        self.num_updates = params["num_updates"]
        self.num_transient = params["num_transient"]

    def tune_thresholds(self, data, iterations=100):
        """
        Allow thresholds to reach stable values by running the network
        with alpha = beta = 0 and gamma set to some small non-zero value
        """
        for _ in range(iterations):
            outputs = self.forward(data)
            self.thresholds += self.gamma * (outputs - self.p)

    def init_transient(self, data, num_iter):
        """
        Estimates y based on diff eq dy/dt
        """
        qx = np.dot(data, self.neurons) # num_points x 16
        y = np.zeros((self.neurons.shape[1], data.shape[0])) # 16 x num_points
        for _ in range(num_iter):
            wy = np.dot(self.inhib, y) # (16 x num_points)
            dydt = qx.T + wy - self.thresholds # (16 x num_points)
            dydt = sigmoid(dydt, self.lamb) - y # (16 x num_points)
            y += self.epsilon * dydt
        output = np.zeros((qx.shape[1], qx.shape[0])) # 16 x num_points
        output[y > 0.5] = 1
        return output

    def update_weights(self, data, y):
        yiyj = np.dot(y, y.T) / data.shape[0]
        y_agg = np.mean(y, axis=1)

        # Update thresholds
        dt = np.reshape(self.gamma * (y_agg - self.p), self.thresholds.shape)
        self.thresholds += dt

        # Update inhibitory weights
        d_inhib = -self.alpha * (yiyj - pow(self.p, 2))
        self.inhib += d_inhib

        # Constrain weights 
        self.inhib[self.inhib > 0] = 0
        for i in range(self.inhib.shape[0]):
            self.inhib[i, i] = 0

        # Update forward weights
        yx = np.dot(y, data) / data.shape[0] # (16 x 64)
        yq = self.neurons * y_agg # 64 x 16
        d_neurons = self.beta * (yx.T - yq)
        self.neurons += d_neurons # 64 x 16

    def train(self, gen):
        """
        `gen` is a function that creates batches of bar data
        """
        for x in range(self.num_updates):
            data = gen(100, int(self.input_dims**0.5), self.p)
            y = self.init_transient(data, self.num_transient)
            self.update_weights(data, y)
        
        self.display_neurons()

    def display_neurons(self):
        fig, ax = plt.subplots(nrows=4, ncols=4)
        r = 0
        for row in ax:
            c = 0
            for col in row:
                col.imshow(np.reshape(self.neurons[:, (r * 4) + c], (8, 8)), cmap="gray")
                c += 1
            r += 1
        plt.show()

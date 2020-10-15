import numpy as np

def sigmoid(x, lamb):
    return 1 / (1 + np.exp(x))

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
        self.inhib = self.zeros(num_neurons, num_neurons)
        self.thresholds = np.reshape(np.ones((num_neurons)), (num_neurons, 1))
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.lamb = params["lamb"]
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
            dydt = sigmoid(dydt) - y # (16 x num_points)
            y += epsilon * dydt
        output = np.zeros((qx.shape[1], qx.shape[0])) # 16 x num_points
        output[y > 0.5] = 1
        return output


    def train(self, data):
        """
        data should be of dimensions (num_examples, pixels)
        In my examples, pixels will be 64 so I can display
        64 pixel images
        """
        for _ in range(self.num_updates):
            y = self.init_transient(data, self.num_transient)
            self.update_weights(data, y)

    def update_weights(self, data, y):
        yiyj = np.dot(y, y.T) / data.shape[0]
        y_agg = np.mean(y, axis=1)

        # Update thresholds
        dt = self.gamma * (y_agg - self.p)
        self.thresholds += dt

        # Update inhibitory weights
        d_inhib = -self.alpha * (yiyj - pow(self.p, 2))
        self.inhibs += d_inhib

        # Constrain weights 
        self.inhibs[self.inhibs > 0] = 0
        for i in range(self.inhibs.shape[0]):
            self.inhibs[i, i] = 0

        # Update forward weights
        yx = np.dot(y, data) / data.shape[0] # (16 x 64)
        yq = self.neurons * y_agg # 64 x 16
        d_neurons = beta * (yx.T - yq) 
        self.neurons += d_neurons # 64 x 16

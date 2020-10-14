import numpy as np

def sigmoid(x, lamb):
    return 1 / (1 + np.exp(-lamb * x))

class Model:
    def __init__(self, num_neurons, input_dims, alpha, beta, gamma, lamb, p):
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
        self.neurons = np.random.randn(input_dims, num_neurons)
        self.inhib = self.zeros(num_neurons, num_neurons)
        self.thresholds = np.random.randn(1, num_neurons)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lamb = lamb
        self.p = p

    def forward(self, data):
        """
        Computes the output of the network based on input
        data. `data` should be (num_examples, pixels)
        """
        output = np.dot(data, self.neurons) # num_examples x num_neurons
        output = sigmoid(output, self.lamb)
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        return output

    def tune_thresholds(self, data, iterations=100):
        """
        Allow thresholds to reach stable values by running the network
        with alpha = beta = 0 and gamma set to some small non-zero value
        """
        for _ in range(iterations):
            outputs = self.forward(data)
            self.thresholds += self.gamma * (outputs - self.p)

    def train(self, data):
        """
        data should be of dimensions (num_examples, pixels)
        In my examples, pixels will be 64 so I can display
        64 pixel images
        """
        outputs = self.forward(data)

        # Update thresholds
        t_grad = (outputs - p) * self.gamma
        self.thresholds += t_grad

        # Update inhibitory weights
        for i in range(self.inhib.shape[0]):
            for j in range(self.inhib.shape[1]):
                if i != j or self.inhib[i, j] > 0:
                    self.inhib[i, j] = 0
                else:
                    grad = -self.alpha * (outputs[i] * outputs[j] - pow(self.p, 2))
                    self.inhib[i, j] += grad

        # Update feed-forward weights



Model(16, 64, 0, 0, 0.1, 10)
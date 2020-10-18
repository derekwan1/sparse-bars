import numpy as np
import matplotlib.pyplot as plt

class PCAModel:
    """
    A model based on Sanger's rule / PCA.
    """
    def __init__(self, weights, epsilon, num_iter, p):
        self.weights = weights # 64 x 16
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.p = p

    def update_weights(self, data):
        output = np.dot(data, self.weights) # (num_samples x 16)
        dw = np.zeros(self.weights.shape) # 64 x 16
        for i in range(self.weights.shape[1]):
            # ith weight * ith neuron | 64 x 100
            # We can re-use previous yi * wi's to calculate
            # each dw_i
            data -= np.dot( \
                    np.reshape(self.weights[:, i], (self.weights.shape[0], 1)), \
                    np.reshape(output.T[i, :], (1, output.shape[0]))).T
            dw[:, i] += np.dot(data.T, output[:, i]) # (64 x 100) x (100 x 1)
        self.weights += dw * self.epsilon

    def train(self, gen):
        self.display_neurons(True)
        for _ in range(self.num_iter):
            data = gen(100, 8, self.p)
            self.update_weights(data)
        self.display_neurons()

    def display_neurons(self, before=False):
        fig, ax = plt.subplots(nrows=4, ncols=4)
        if before:
            fig.suptitle(f"Initial neurons")
        else:
            fig.suptitle(f"Neurons after {self.num_iter} training rounds")
        r = 0
        for row in ax:
            c = 0
            for col in row:
                col.imshow(np.reshape(self.weights[:, (r * 4) + c], (8, 8)), cmap="gray")
                c += 1
            r += 1
        plt.show()

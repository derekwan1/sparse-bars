import model as m
import numpy as np
import matplotlib.pyplot as plt

def generate_bars(num_samples, size, p, verbosity=False):
    """
    Generate `num_samples` of bar images
    that are of dimensions size x size pixels

    Each bar has `p` probability of being "on"
    """
    data = np.zeros((num_samples, size, size))

    # horizontal bars
    for i in range(size):
        indices = np.array([True if np.random.uniform(0, 1) < p else False for _ in range(num_samples)])
        data[indices, i, :] = 1

    # vertical bars
    for i in range(size):
        indices = np.array([True if np.random.uniform(0, 1) < p else False for _ in range(num_samples)])
        data[indices, :, i] = 1

    # display some images
    if verbosity:
        for i in range(5):
            plt.imshow(data[i, :, :], cmap="gray")
            plt.show()

    # vectorize the images
    return np.reshape(data, (num_samples, size * size))

if __name__ == "__main__":
    p = 0.1
    params = {
        "num_neurons": 16,
        "input_dims": 64,
        "alpha": 0.1,
        "beta": 0.05,
        "gamma": 0.1,
        "lamb": 10,
        "epsilon": 0.001,
        "p": p,
        "num_updates": 1000,
        "num_transient": 1000
    }
    model = m.Model(params)
    model.train(generate_bars)
    # generate_bars(100, 8, p)

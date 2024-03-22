import random
import numpy as np


def corrupt_labels(y_train, random_label_noise=0.0):
    random.seed(42)
    np.random.seed(42)

    y_train = np.array(y_train, copy=True)
    possible_labels = np.unique(y_train)

    # Choose n random indices
    n_corrupted = int(len(y_train) * random_label_noise)
    random_noise_indices = np.random.choice(
        len(y_train), size=n_corrupted, replace=False)
    random_noise = np.full(y_train.shape, True)
    random_noise[random_noise_indices] = False

    # Generate alternate version of labels where every single one is corrupted
    # (Draws randomly from all values EXCEPT true value)
    def gen_corrupt_label(_, i):
        i = int(i)
        from_labels = possible_labels[possible_labels != y_train[i]].tolist()
        return random.sample(from_labels, 1)[0]
    corrupted_labels = np.fromfunction(
        np.vectorize(gen_corrupt_label), (1, len(y_train)))[0]

    return np.where(random_noise, y_train, corrupted_labels)


if __name__ == '__main__':
    b = np.array([0, 1, 1, 1, 1, 1, 1, 1, 2])
    print(b)
    print(corrupt_labels(b, 0.5))

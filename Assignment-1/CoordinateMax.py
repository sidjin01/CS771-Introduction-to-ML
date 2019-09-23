#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from time import perf_counter

def loss(W, C, Y, X):
    out = Y * np.matmul(X, W)
    hinge = np.maximum(0, 1 - out)
    cost = np.matmul(W[:-1].T, W[:-1]) + C * np.sum(hinge ** 2)
    return np.squeeze(cost)


def SCM(X, Y):
    b = np.ones((len(X), 1))
    X = np.concatenate([X, b], 1)
    y = np.squeeze(Y)

    C = 1.0
    epochs = 5
    lr = 1.0
    decay_rate = 1.5
    decay_eps = 1

    np.random.seed(42)
    alpha = np.random.uniform(size=(len(X), 1))
    W = np.matmul(X.T, Y * alpha)

    loss_series = []
    time_series = []
    tic = perf_counter()

    with tqdm(range(1, epochs + 1), desc="Optimizing") as pbar:
        for ep in pbar:
            net_loss = 0

            for i, coord in enumerate(np.random.permutation(len(X)), 1):
                alpha_new = 2 * C * (1 - y[coord] * np.matmul(X[coord:(coord + 1)], W))
                alpha_new = np.squeeze(np.clip(alpha_new, 0, C))
                alpha[coord] += lr * (alpha_new - alpha[coord])

                W = np.matmul(X.T, Y * alpha)
                net_loss += loss(W, C, Y, X)
                pbar.set_postfix({"loss": np.round(net_loss / i, 2)})

                loss_series.append(net_loss / i)
                time_series.append(perf_counter() - tic)

            if ep % decay_eps == 0:
                lr /= decay_rate

    # y = np.matmul(X, W)
    # y = np.int32(np.where(y > 0, 1, -1))
    # accuracy = np.mean(np.int32(y == Y))
    # print(f"Accuracy: {round(accuracy * 100, 2)}%")

    return time_series, loss_series


def main():
    Z = np.loadtxt("/Users/siddjin/Desktop/assn1/data")
    y = Z[:, 0]
    Y = np.expand_dims(y, 1)
    X = Z[:, 1:]
    SCM(X, Y)


if __name__ == "__main__":
    main()
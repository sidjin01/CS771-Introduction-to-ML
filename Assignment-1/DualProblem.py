#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from time import perf_counter


def loss(W, C, X, Y):
    out = Y * np.matmul(X, W)
    hinge = np.maximum(0, 1 - out)
    cost = np.matmul(W[:-1].T, W[:-1]) + C * np.sum(hinge ** 2)
    return np.squeeze(cost)



def minibatchGA(X, Y, C, alpha, lr):
    alpha_grad = 1 - alpha / (2 * C) - Y * np.matmul(X, np.matmul(X.T, Y * alpha))
    alpha += lr * alpha_grad
    alpha = np.clip(alpha, 0, C)
    return alpha


def dual_problem(X, Y):
    b = np.ones((len(X), 1))
    X = np.concatenate([X, b], 1)

    C = 1.0
    lr = 5e-5
    batch_size = 8192
    epochs = 5000

    np.random.seed(42)
    alpha = np.random.uniform(size=(len(X), 1))

    loss_series = []
    time_series = []
    tic = perf_counter()

    with tqdm(range(epochs), desc="Training") as pbar:
        for _ in pbar:
            net_loss = 0

            for i, start in enumerate(range(0, len(X), batch_size), 1):
                batch_X = X[start:(start + batch_size)]
                batch_Y = Y[start:(start + batch_size)]
                batch_alpha = alpha[start:(start + batch_size)]

                batch_alpha = minibatchGA(batch_X, batch_Y, C, batch_alpha, lr)
                alpha[start:(start + batch_size)] = batch_alpha
                W = np.matmul(batch_X.T, batch_Y * batch_alpha)
                net_loss += loss(W, C, X, Y)
                pbar.set_postfix({"loss": np.round(net_loss / i, 2)})

                loss_series.append(net_loss / i)
                time_series.append(perf_counter() - tic)

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
    dual_problem(X, Y)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from time import perf_counter


def loss(X, Y, W, b, C):
    out = Y * (np.matmul(X, W) + b)
    hinge = np.maximum(0, 1 - out)
    cost = np.matmul(W.T, W) + C * np.sum(hinge ** 2)
    return np.squeeze(cost)


def minibatchSGD(X, Y, C, W, b, lr):
    out = Y * (np.matmul(X, W) + b)
    hinge = np.maximum(0, 1 - out)
    der = np.where(out < 1, -1.0, 0.0)

    W_grad = W + 2 * C * np.matmul(X.T, hinge * der * Y) 
    b_grad = 2 * C * np.sum(hinge * der * Y, axis=0)
    W -= lr * W_grad
    b -= lr * b_grad
    return W, b


def mini_batch_gd(X, Y):
    C = 1.0
    lr = 1e-5
    batch_size = 8192
    epochs = 5000

    np.random.seed(42)
    b = np.random.rand(1)
    W = np.random.normal(size=(X.shape[1], 1))

    loss_series = []
    time_series = []
    tic = perf_counter()

    with tqdm(range(epochs), desc="Training") as pbar:
        for _ in pbar:
            net_loss = 0

            for i, start in enumerate(range(0, len(X), batch_size), 1):
                batch_X = X[start:(start + batch_size)]
                batch_Y = Y[start:(start + batch_size)]

                W, b = minibatchSGD(batch_X, batch_Y, C, W, b, lr)
                net_loss += loss(X, Y, W, b, C)
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
    mini_batch_gd(X, Y)


if __name__ == "__main__":
    main()
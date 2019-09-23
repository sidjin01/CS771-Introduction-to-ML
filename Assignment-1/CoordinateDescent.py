#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
from time import perf_counter


def loss(X, Y, W, C):
    out = Y * np.matmul(X, W)
    hinge = np.maximum(0, 1 - out)
    cost = np.matmul(W.T, W) + C * np.sum(hinge ** 2)
    return np.squeeze(cost)


def GD(X, Y, C, W, lr):
    out = Y * np.matmul(X, W)
    der = np.where(out < 1, -1.0, 0.0)
    hinge = np.maximum(0, 1 - out)
    
    W_grad = W + 2 * C * np.matmul(X.T, hinge * der * Y)
    W = W - lr * W_grad
    return W


def coord_descent(X, Y):
    b = np.ones((len(X), 1))
    X = np.concatenate([X, b], 1)

    C = 1.0
    alpha = 1e-5 
    epochs = 2000

    np.random.seed(42)
    W = np.random.normal(size=(X.shape[1], 1))

    loss_series = []
    time_series = []
    tic = perf_counter()

    with tqdm(range(epochs), desc="Training") as pbar:
        for _ in pbar:
            net_loss = 0

            for i, coord in enumerate(np.random.permutation(X.shape[1]), 1):
                W[coord] = GD(X, Y, C, W, alpha)[coord]
                net_loss += loss(X, Y, W, C)
                pbar.set_postfix({"loss": net_loss / i})

                loss_series.append(net_loss / i)
                time_series.append(perf_counter() - tic)

    # y = np.matmul(X, W)
    # y = np.int32(np.where(y > 0, 1, -1))
    # accuracy = np.mean(np.int32(y == Y))
    # print(f"Accuracy: {round(accuracy * 100, 2)}%")

    return time_series, loss_series


def main():
    Z = np.loadtxt( "/Users/siddjin/Desktop/assn1/data" )
    y = Z[:, 0]
    Y = np.expand_dims(y, 1)
    X = Z[:, 1:]
    coord_descent(X, Y)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from MiniBatchGD import mini_batch_gd
from DualProblem import dual_problem
from CoordinateMax import SCM


def plotter(X, Y):
    sns.set()
    plt.ylim(0, 1e5)

    ax = sns.lineplot(*mini_batch_gd(X, Y), label="Mini-batch SGD on P1")
    ax = sns.lineplot(*dual_problem(X, Y), label="Projected Mini-batch SGA on D2")
    ax = sns.lineplot(*SCM(X, Y), label="Coordinate Maximization on D2")

    ax.set_title("Convergence Curves")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Loss (P1)")

    plt.savefig("convergence-curves.png", bbox_inches="tight")
    plt.show()


def main():
    Z = np.loadtxt("/Users/siddjin/Desktop/assn1/data")
    y = Z[:, 0]
    Y = np.expand_dims(y, 1)
    X = Z[:, 1:]
    plotter(X, Y)


if __name__ == "__main__":
    main()
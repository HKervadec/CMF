#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def plot_2d(λ, erriter):
    fig, axes = plt.subplots(2)

    axes[0].imshow(λ)
    axes[0].set_title("Segmentation result λ")

    axes[1].plot(erriter)
    axes[1].set_title("abs(errλ) over i")

    plt.show()


def plot_3d(λ, erriter):
    fig, axes = plt.subplots(1)

    axes.plot(erriter)
    axes.set_title("abs(errλ) over i")

    plt.show()
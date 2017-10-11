#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt


def CMF_2D(ur, params):
    rows, cols, numIter, errbound, cc, steps = params
    imgSize = ur.size
    assert(rows*cols == imgSize)

    α = .5 * np.ones((rows, cols))

    # Build the data terms
    ulab = [0.15, 0.6]
    Cs = np.abs(ur - ulab[0])
    Ct = np.abs(ur - ulab[1])

    # set the initial values
    # - the initial value of λ is set to be an initial cut, see below.
    # - the initial values of two terminal flows ps and pt are set to be the
    # specified legal flows.
    # - the initial value of the spatial flow fiels p = (pp1, pp2) is set to
    # be zero.
    λ = np.asarray((Cs - Ct) >= 0, np.float64)
    ps = np.minimum(Cs, Ct)  # Minimum element wise between the two
    pt = ps

    pp1 = np.zeros((rows, cols+1))
    pp2 = np.zeros((rows+1, cols))
    divp = np.zeros((rows, cols))

    erriter = np.zeros(numIter)

    for i in range(numIter):
        # update the spatial flow field p = (pp1, pp2):
        # the following steps are the gradient descent step with steps as the
        # step-size.
        pts = divp - (ps - pt + λ/cc)
        pp1[:, 1:-1] += steps * (pts[:, 1:] - pts[:, :-1])
        pp2[1:-1, :] += steps * (pts[1:, :] - pts[:-1, :])

        # the following steps give the projection to make |p(x)| <= α(x)
        squares = pp1[:, :-1]**2 + pp1[:, 1:]**2 + pp2[:-1, :]**2 + pp2[1:, :]**2
        gk = np.sqrt(squares * .5)
        gk = (gk <= α) + np.logical_not(gk <= α) * (gk / α)
        gk = 1 / gk

        pp1[:, 1:-1] = (.5 * (gk[:, 1:] + gk[:, :-1])) * (pp1[:, 1:-1])
        pp2[1:-1, :] = (.5 * (gk[1:, :] + gk[:-1, :])) * (pp2[1:-1, :])

        divp = pp1[:, 1:] - pp1[:, :-1] + pp2[1:, :] - pp2[:-1, :]

        # updata the source flow ps
        pts = divp + pt - λ/cc + 1/cc
        ps = np.minimum(pts, Cs)

        # update the sink flow pt
        pts = - divp + ps + λ/cc
        pt = np.minimum(pts, Ct)

        errλ = cc * (divp + pt - ps)
        λ -= errλ

        erriter[i] = np.sum(np.abs(errλ)) / imgSize

        if erriter[i] < errbound:
            return λ, erriter, i

    return λ, erriter, numIter


def plot(λ, erriter):
    fig, axes = plt.subplots(2)

    axes[0].imshow(λ)
    axes[0].set_title("Segmentation result λ")

    axes[1].plot(erriter)
    axes[1].set_title("abs(errλ) over i")

    plt.show()


if __name__ == "__main__":
    im_file = "../data/cameraman.jpg"

    ur = sp.misc.imread(im_file) / 255
    plt.imshow(ur)
    rows, cols = ur.shape

    varParas = [rows, cols, 300, 1e-4, 0.3, 0.16]
    """
    para 0,1 - rows, cols of the given image
    para 2 - the maximum number of iterations
    para 3 - the error bound for convergence
    para 4 - cc for the step-size of augmented Lagrangian method
    para 5 - the step-size for the graident-projection of p
    """

    λ, erriter, i = CMF_2D(ur, varParas)
    print("Iterations: {}".format(i))
    print("Erriter mean: {}".format(np.mean(erriter)))
    print("λ mean: {}".format(np.mean(λ)))

    plot(λ, erriter[:i])

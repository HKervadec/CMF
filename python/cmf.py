#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.misc


def CMF(ur, params):
    rows, cols, numIter, errbound, cc, steps = params
    imgSize = ur.size
    assert(rows*cols == imgSize)

    α = .5 * np.ones((rows, cols))

    # Build the data terms
    ulab = [0.15, 0.6]
    Cs = np.abs(ur - ulab[0])
    Ct = np.abs(ur - ulab[1])

    # set the initial values
    # - the initial value of u is set to be an initial cut, see below.
    # - the initial values of two terminal flows ps and pt are set to be the
    # specified legal flows.
    # - the initial value of the spatial flow fiels p = (pp1, pp2) is set to
    # be zero.
    u = np.asarray((Cs - Ct) >= 0, np.float64)
    ps = np.minimum(Cs, Ct)  # Minimum element wise between the two
    pt = ps

    pp1 = np.zeros((rows, cols+1))
    pp2 = np.zeros((rows+1, cols))
    # divp = pp1[:, 1:] - pp1[:, :cols] + pp2[1:, :] - pp2[:rows, :]
    divp = np.zeros((rows, cols))

    erriter = np.zeros(numIter)

    for i in range(numIter):
        # update the spatial flow field p = (pp1, pp2):
        # the following steps are the gradient descent step with steps as the
        # step-size.
        pts = divp - (ps - pt + u/cc)
        pp1[:, 1:cols] += steps * (pts[:, 1:cols] - pts[:, :cols-1])
        pp2[1:rows, :] += steps * (pts[1:rows, :] - pts[:rows-1, :])

        # the following steps give the projection to make |p(x)| <= α(x)
        squares = pp1[:, :cols]**2  + pp1[:, 1:]**2 + pp2[:rows, :]**2 + pp2[1:, :]**2
        gk = np.sqrt(squares * .5)
        gk = (gk <= α) + np.logical_not(gk <= α) * (gk / α)
        gk = 1 / gk

        pp1[:, 1:cols] = (.5 * (gk[:, 1:cols] + gk[:, :cols-1])) * (pp1[:, 1:cols])
        pp2[1:rows, :] = (.5 * (gk[1:rows, :] + gk[:rows-1, :])) * (pp2[1:rows, :])

        # updata the source flow ps
        pts = divp + pt - u/cc + 1/cc
        ps = np.minimum(pts, Cs)

        # update the sink flow pt
        pt = - divp + ps + u/cc
        pt = np.minimum(pts, Ct)

        erru = cc * (divp + pt - ps)
        u -= erru

        erriter[i] = np.sum(np.abs(erru)) / imgSize

        if erriter[i] < errbound:
            return u, erriter, i

    return u, erriter, numIter


if __name__ == "__main__":
    im_file = "../data/cameraman.jpg"

    ur = sp.misc.imread(im_file)
    ur = ur / 255
    rows, cols = ur.shape

    varParas = [rows, cols, 300, 1e-4, 0.3, 0.16]
    """
    para 0,1 - rows, cols of the given image
    para 2 - the maximum number of iterations
    para 3 - the error bound for convergence
    para 4 - cc for the step-size of augmented Lagrangian method
    para 5 - the step-size for the graident-projection of p
    """

    u, erriter, i = CMF(ur, varParas)

    print("toto")

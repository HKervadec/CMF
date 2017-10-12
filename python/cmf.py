#!/usr/bin/env python3

import numpy as np


def CMF_2D(ur, params):
    numIter, errbound, cc, steps = params
    rows, cols = ur.shape
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


def CMF_3D(ur, params):
    numIter, errbound, cc, steps = params
    rows, cols, height = ur.shape
    imgSize = ur.size
    assert(rows*cols*height == imgSize)

    α = .5 * np.ones((rows, cols, height))

    # Build the data terms
    ulab = [0.2, 0.7]
    Cs = np.abs(ur - ulab[0])
    Ct = np.abs(ur - ulab[1])

    # set the initial values
    λ = np.asarray((Cs - Ct) >= 0, np.float64)
    ps = np.minimum(Cs, Ct)  # Minimum element wise between the two
    pt = ps

    pp1 = np.zeros((rows, cols+1, height))
    pp2 = np.zeros((rows+1, cols, height))
    pp3 = np.zeros((rows, cols, height+1))
    divp = np.zeros((rows, cols, height))

    erriter = np.zeros(numIter)

    for i in range(numIter):
        # update the spatial flow field p = (pp1, pp2):
        # the following steps are the gradient descent step with steps as the
        # step-size.
        pts = divp - (ps - pt + λ/cc)
        pp1[:, 1:-1, :] += steps * (pts[:, 1:, :] - pts[:, :-1, :])
        pp2[1:-1, :, :] += steps * (pts[1:, :, :] - pts[:-1, :, :])
        pp3[:, :, 1:-1] += steps * (pts[:, :, 1:] - pts[:, :, :-1])

        # the following steps give the projection to make |p(x)| <= α(x)
        squares =  pp1[:, :-1, :]**2 + pp1[:, 1:, :]**2
        squares += pp2[:-1, :, :]**2 + pp2[1:, :, :]**2
        squares += pp3[:, :, :-1]**2 + pp3[:, :, 1:]**2
        gk = np.sqrt(squares * .5)
        gk = (gk <= α) + np.logical_not(gk <= α) * (gk / α)
        gk = 1 / gk

        pp1[:, 1:-1, :] = (.5 * (gk[:, 1:, :] + gk[:, :-1, :])) * (pp1[:, 1:-1, :])
        pp2[1:-1, :, :] = (.5 * (gk[1:, :, :] + gk[:-1, :, :])) * (pp2[1:-1, :, :])
        pp3[:, :, 1:-1] = (.5 * (gk[:, :, 1:] + gk[:, :, :-1])) * (pp3[:, :, 1:-1])

        divp =  pp1[:, 1:, :] - pp1[:, :-1, :]
        divp += pp2[1:, :, :] - pp2[:-1, :, :]
        divp += pp3[:, :, 1:] - pp3[:, :, :-1]

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

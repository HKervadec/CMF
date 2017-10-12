#!/usr/bin/env python3

from sys import argv
import numpy as np
import scipy as sp
import scipy.misc
import nibabel as nib

from plot import plot_2d, plot_3d
from cmf import CMF_2D, CMF_3D


if __name__ == "__main__":
    if len(argv) > 1:
        target = argv[1]
    else:
        target = "2"

    if target == "2":
        im_file = "../data/cameraman.jpg"
        ur = sp.misc.imread(im_file) / 255
        # plt.imshow(ur)

        varParas = [300, 1e-4, 0.3, 0.16]
        """
        para 0 - the maximum number of iterations
        para 1 - the error bound for convergence
        para 2 - cc for the step-size of augmented Lagrangian method
        para 3 - the step-size for the graident-projection of p
        """

        λ, erriter, i = CMF_2D(ur, varParas)
        print("Iterations: {}".format(i))
        print("Erriter mean: {}".format(np.mean(erriter)))
        print("λ mean: {}".format(np.mean(λ)))

        plot_2d(λ, erriter[:i])
    elif target == "3":
        im_file = "../data/IM_0190_frame_01.nii"
        ur = np.asarray(nib.load(im_file).dataobj) / 255

        varParas = [200, 5e-4, 0.35, 0.11]
        """
        para 0 - the maximum number of iterations
        para 1 - the error bound for convergence
        para 2 - cc for the step-size of augmented Lagrangian method
        para 3 - the step-size for the graident-projection of p
        """
        λ, erriter, i = CMF_3D(ur, varParas)

        print("Iterations: {}".format(i))
        print("Erriter mean: {}".format(np.mean(erriter)))
        print("λ mean: {}".format(np.mean(λ)))
        print("λ shape: {}".format(λ.shape))

        plot_3d(λ, erriter)
    else:
        raise Exception("Error: example not found")
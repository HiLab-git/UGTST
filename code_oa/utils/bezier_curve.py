import numpy as np
import random
import torch
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=0.5):
    # Note that this function will not help you to do normalization.
    # Once it normalizes the image, it would transform back after the nonlinear transformation.
    if random.random() >= prob:
        return x

    maxvalue = x.max()
    minvalue = x.min()

    # Normalize to [0, 1] using max-min normalization
    if maxvalue > 1 or minvalue < 0:
        # Normalize x to [0, 1]
        x_maxmin = (x - minvalue) / torch.clamp((maxvalue - minvalue), min=1e-5)

    # Do nonlinear transformation (Bezier curve)
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)

    if random.random() < 0.5:
        # Half chance to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    # Convert x_maxmin to numpy (it should be on the CPU at this point)
    x_maxmin_cpu = x_maxmin.cpu().numpy()  # Move the tensor to CPU and convert to numpy

    # Perform interpolation using numpy
    nonlinear_x = np.interp(x_maxmin_cpu, xvals, yvals)

    # Convert nonlinear_x back to a tensor, and ensure it's on the same device as x
    nonlinear_x = torch.tensor(nonlinear_x, dtype=torch.float32,
                               device=x.device)  # Ensure it is on the same device as x

    # Restore the original intensity from the MAXMIN normalization
    nonlinear_x = nonlinear_x * torch.clamp((maxvalue - minvalue), min=1e-5) + minvalue

    return nonlinear_x
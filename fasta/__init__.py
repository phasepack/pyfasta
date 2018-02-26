"""A Python implementation of Goldstein et. al's Fast Adaptive Shrinkage/Thresholding Algorithm for convex optimization.

FASTA minimizes a function expressed in the form,

    min h(x) = f(Ax) + g(x),

where `f` is convex and differentiable and `g` is convex.
`g` may be non-differentiable, or possibly not even finite-valued, so normal gradient descent methods cannot be used.

The FASTA algorithm is a variant of the Forward-Backward Splitting (FBS) algorithm, which alternatively takes steps
to optimize `h` by optimizing `f` and then `g`. The optimization of `g` relies on the computation of the proximal
operator of `g` with stepsize `t`, equal to argmin { t*g(x) + .5 ||x-z||**2 }, which finds a point `x` close to the
minimum value of `g` while not straying too far from `z`.

The FASTA algorithm incorporates various improvements to FBS, as detailed in (Goldstein et al. 2016), including
adaptive stepsize selection, acceleration, non-monotone backtracking line search,
and a variety of stopping conditions. Additionally, included in this pacakage are eleven different example problems that
are solved using FASTA.
"""

import numpy as np
from numpy import linalg as la
from time import time
from typing import Callable

from . import proximal, stopping, linalg

__author__ = "Noah Singer"

__all__ = ["fasta", "Convergence"]

EPSILON = 1E-12


# TODO: check mutually allowed modes
# TODO: adjust to allow tensors

def fasta(A: linalg.LinearMap,
          f: Callable[[np.ndarray], float], gradf: Callable[[np.ndarray], np.ndarray],
          g: Callable[[np.ndarray], float], proxg: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,

          adaptive: bool=True, accelerate: bool=False, verbose: bool=True,

          max_iters: int=1000, tolerance: float=1e-5,
          stop_rule: Callable[[int, float, float, float, float], bool]=stopping.hybrid_residual,

          L: float=None, tau0: float=None,

          backtrack: bool=True, stepsize_shrink: bool=None, window: int=10, max_backtracks: int=20,
          restart: bool=True,

          evaluate_objective: bool=False, record_iterates: bool=False,
          func: Callable[[np.ndarray], np.ndarray]=None) -> "Convergence":
    """Run the FASTA algorithm.

    :param adaptive: Adaptively choose the stepsize by locally approximating the function as a quadratic (default: True)
    :param accelerate: Increase the stepsize at every step of the algorithm (default: False)
    :param verbose: Print detailed convergence information as the algorithm progresses (default: False)
    :param A: A linear operator (often just a matrix)
    :param At: The adjoint (conjugate transpose) of A
    :param f: A convex, differentiable function of x
    :param gradf: The gradient of f
    :param g: A convex function of x
    :param proxg: The proximal operator of g with stepsize t
    :param x0: An initial guess for position of the optimal value (often a vector of zeroes)
    :param max_iters: The maximum number of iterations allowed by the algorithm (default: 1000)
    :param tolerance: The numerical tolerance of the algorithm (default: 1e-3)
    :param stop_rule: A function that checks whether the algorithm should terminate (default: stopping.hybrid_residual)
    :param L: The Lipschitz constant of f (default: the term is approximated). Only required if tau is not set
    :param tau0: The initial stepsize for the algorithm (default: computed from L)
    :param backtrack: Use backtracking line search (default: True)
    :param stepsize_shrink: When backtracking, decrease the stepsize to prevent further mistakes (default: 0.2 when backtracking, 0.5 otherwise)
    :param window: The lookback window for backtracking (default: 10)
    :param max_backtracks: The maximum total number of backtracks allowed in a single iteration of the algorithm (default: 20)
    :param restart: Restart the acceleration of FISTA. Only relevant when accelerating (default: True)
    :param evaluate_objective: Whether to evaluate the quality of each iterate by the value of the objective (and also record the objective at every step). Otherwise, the iterate quality is judged by the residual (default: False)
    :param record_iterates: Whether to record the iterate after each iteration (default: False)
    :param func: A scalar function to evaluate after each iteration (default: None)
    :return: A guess at an optimizer of h
    """

    # Conventions:
    #   - Variables ending with 0 indicate the previous round's values
    #   - Variables ending with 1 indicate the current round's values
    #   - Variables ending with _hist indicate a history that is tracked between rounds

    # Option to just do gradient descent
    if g is None:
        g = lambda x: 0
        proxg = lambda x, t: x

    if stepsize_shrink is None and backtrack:
        if adaptive:
            # This is more aggressive, since the stepsize increases dynamically
            stepsize_shrink = 0.2
        else:
            stepsize_shrink = 0.5

    # Check if we need to approximate the Lipschitz constant of f
    if not L or not tau0:
        # Compute two random vectors
        x1 = np.random.randn(*x0.shape)
        x2 = np.random.randn(*x0.shape)

        # Compute the gradients between the vectors
        gradf1 = A.H(gradf(A(x1)))
        gradf2 = A.H(gradf(A(x2)))

        # Approximate the Lipschitz constant of f
        L = la.norm((gradf1 - gradf2).ravel()) / la.norm((x1 - x2).ravel())

        # We're guaranteed that FBS converges for tau <= 2.0 / L
        tau0 = (2 / L) / 10

    if not tau0:
        tau0 = 1 / L

    if verbose:
        print("Initializing FASTA...\n")
        print("Iteration #\tResidual\tStepsize\tAccel. param\tBacktracks\tObjective")

    # Allocate memory for convergence information
    residual_hist = np.zeros(max_iters)
    norm_residual_hist = np.zeros(max_iters)
    tau_hist = np.zeros(max_iters)
    f_hist = np.zeros(max_iters+1)
    times = np.zeros(max_iters+1)

    total_backtracks = 0

    # Initialize values
    x1 = x0
    tau1 = tau0

    z1 = A(x1)
    f1 = f(z1)
    gradf1 = A.H(gradf(z1))

    f_hist[0] = f1

    if evaluate_objective:
        objective_hist = np.zeros(max_iters+1)
        objective_hist[0] = f1 + g(x1)

    if record_iterates:
        iterate_hist = np.zeros((max_iters + 1,) + x0.shape)
        iterate_hist[0] = x1

    if func:
        function_hist = np.zeros(max_iters+1)
        function_hist[0] = func(x1)

    # Additional initialization for accelerative descent
    if accelerate:
        x_accel1 = x1
        z_accel1 = z1
        alpha1 = 1.0

    # Additional initialization for backtracking
    if backtrack:
        total_backtracks = 0

    # Stopping conditions may be monotonic, so we always want to take the best iterate
    # Quality is evaluated as lowest objective, when objective is evaluated, or smallest residual, when it's not
    max_residual = -np.inf
    best_quality = np.inf
    best_iterate = x0

    # Algorithm loop
    i = 0
    while i < max_iters:
        # Start timing this iteration
        times[i] = time()

        # Rename last iteration's current variables to this round's former variables
        x0 = x1
        gradf0 = gradf1
        tau0 = tau1

        # Perform FBS: Take the forwards step by moving x in the direction of f's gradient0
        x1hat = x0 - tau0 * gradf1

        # Now take the backwards step by finding a minimizer of g close to x
        x1 = proxg(x1hat, tau0)

        Dx = x1 - x0
        z1 = A(x1)
        f1 = f(z1)

        # Track the number of total backtracks
        backtrack_count = 0

        # Non-monotone backtracking line search, used to guarantee convergence and balance out adaptive search if
        # stepsizes grow too large
        if backtrack:
            # Find the maximum of the last `window` values of f
            M = np.max(f_hist[max(i-window+1, 0):(i+1)])

            # Check if the quadratic approximation of f is an upper bound; if it's not, FBS isn't guaranteed to converge
            while f1 - (M + np.real(Dx.ravel().T @ gradf0.ravel()) + la.norm(Dx.ravel())**2 / (2 * tau0)) > EPSILON \
                    and backtrack_count < max_backtracks:
                # We've gone too far, so shrink the stepsize and try FBS again (be twice as aggressive for
                # adaptive stepsize selection)
                tau0 *= stepsize_shrink

                # Redo the FBS step
                x1hat = x0 - tau0 * gradf0
                x1 = proxg(x1hat, tau0)

                # Recalculate values
                Dx = x1 - x0
                z1 = A(x1)
                f1 = f(z1)

                backtrack_count += 1

            total_backtracks += backtrack_count

        # FISTA-style acceleration, which works well for ill-conditioned problems
        if accelerate:
            # Rename last round's current variables to this round's previous variables
            x_accel0 = x_accel1
            z_accel0 = z_accel1

            x_accel1 = x1
            z_accel1 = z1

            alpha0 = alpha1

            # Prevent alpha from growing too large by restarting the acceleration
            if restart and (x0 - x1).ravel().T @ (x1 - x_accel0).ravel() > 1E-30:
                alpha0 = 1.0

                if verbose:
                    print("Restarted acceleration.")

            # Recalculate acceleration parameter
            alpha1 = (1 + np.sqrt(1 + 4 * alpha0**2)) / 2

            # Overestimate the next value of x by a factor of (alpha0 - 1) / alpha
            # NOTE: this makes a copy of x1, which is necessary since x1's reference is linked to x0
            x1 = x1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            z1 = z1 + (alpha0 - 1) / alpha1 * (z_accel1 - z_accel0)

            f1 = f(z1)

        # Compute the next iteration's gradient
        gradf1 = A.H(gradf(z1))
        tau1 = tau0

        # Adaptive adjustments of stepsize using the Barzilai-Borwein method (spectral method), which
        # approximates the function as a simple quadratic form, and dynamically selects a stepsize for each iteration
        if adaptive:
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = np.real(Dx.ravel().T @ Dg.ravel())

            # One least squares estimate of the best stepsize
            tau_s = la.norm(Dx.ravel()) ** 2 / dotprod
            # A different least squares estimate of the best stepsize
            tau_m = max(dotprod / la.norm(Dg.ravel()) ** 2, 0)

            # Use an adaptive combination of tau_s and tau_m
            if 2 * tau_m > tau_s:
                tau1 = tau_m
            else:
                tau1 = tau_s - .5 * tau_m

            # Ensure non-negative stepsize
            if tau1 <= 0 or np.isinf(tau1) or np.isnan(tau1):
                tau1 = tau0 * 1.5

        residual_hist[i] = la.norm(Dx.ravel()) / tau0

        normalizer = max(la.norm(gradf0.ravel()), la.norm((x1 - x1hat).ravel()) / tau0) + EPSILON

        # Record convergence information
        tau_hist[i] = tau0
        norm_residual_hist[i] = residual_hist[i] / normalizer
        f_hist[i+1] = f1

        max_residual = max(max_residual, residual_hist[i])

        # If the objective is evaluated, we can find the best iterate using the objective
        if evaluate_objective:
            objective_hist[i+1] = f1 + g(x1)
            quality = objective_hist[i+1]
        # Otherwise, we find the best iterate using the smallest residual
        else:
            quality = residual_hist[i]

        if record_iterates:
            iterate_hist[i+1,...] = x1

        # If we have a function to evaluate, evaluate it
        if func:
            function_hist[i+1] = func(x1)

        if quality < best_quality:
            best_iterate = x1
            best_quality = quality

        if verbose:
            print("[{:<6}]\t{:e}\t{:e}\t{:e}\t{:6}\t{:e}".format(i, residual_hist[i], tau_hist[i],
                                                                   alpha0 if accelerate else 0.0,
                                                                   backtrack_count if backtrack else 0,
                                                                   objective_hist[i] if evaluate_objective else 0))

        if stop_rule(i, residual_hist[i], norm_residual_hist[i], max_residual, tolerance):
            i += 1
            break

        i += 1

    # Record the time at algorithm stop
    times[i] = time()

    return Convergence(residual_hist, norm_residual_hist, tau_hist, total_backtracks, times, i, best_iterate,
                       objective_hist if evaluate_objective else None,
                       iterate_hist if record_iterates else None,
                       function_hist if func else None)


class Convergence:
    """Convergence information about the FASTA algorithm."""

    def __init__(self, residuals: np.ndarray, norm_residuals: np.ndarray, stepsizes: np.ndarray, backtracks: np.ndarray,
                 times: np.ndarray, iteration_count: np.ndarray, solution: np.ndarray, objectives: np.ndarray = None,
                 iterates: np.ndarray = None, function_hist: np.ndarray = None):
        """Record convergence information about FASTA.

        :param residuals: The residuals, or the size differences between iterates, at each step
        :param norm_residuals: The normalized residuals at each step
        :param stepsizes: The stepsizes at each step
        :param backtracks: The number of backtracks performed at each step
        :param times: The time after each iteration is completed. The first entry is before the algorithm starts
        :param iteration_count: The number of iterations until the algorithm converged
        :param solution: The solution the algorithm computed
        :param objectives: The value of the objective function at each step (default: None)
        :param iterates: The
        :param function_hist:
        """
        self.residuals = residuals
        self.norm_residuals = norm_residuals
        self.stepsizes = stepsizes
        self.backtracks = backtracks
        self.times = times
        self.iteration_count = iteration_count
        self.solution = solution
        self.objectives = objectives
        self.iterates = iterates
        self.function_hist = function_hist

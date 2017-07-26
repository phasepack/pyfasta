"""Solve the L1-penalized least squares problem (also known as basis pursuit denoising, or BPDN), min_x mu||x||_1 + .5||Ax-b||^2, using the FASTA solver."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes, NO_ARGS

__author__ = "Noah Singer"

__all__ = ["SparseLeastSquaresProblem"]


class SparseLeastSquaresProblem(ExampleProblem):
    def __init__(self, A, At, b, mu, x=None):
        """Create an instance of the sparse least squares problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param At: The Hermitian adjoint operator of A (for real matrices A, just the transpose)
        :param b: The observation vector
        :param mu: The regularization parameter
        :param x: The true value of the unknown signal, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.b = b
        self.mu = mu
        self.x = x

    @staticmethod
    def construct(M=200, N=1000, K=10, sigma=0.01, mu=0.02):
        """Construct a sample sparse least squares problem with a random sparse signal and measurement matrix.

        :param M: The number of measurements (default: 200)
        :param N: The dimension of the sparse signal (default: 1000)
        :param K: The signal sparsity (default: 10)
        :param sigma: The noise level in the observation vector (default: 0.01)
        :param mu: The regularization parameter (default: 0.02)
        """
        # Create sparse signal
        x = np.zeros(N)
        x[np.random.permutation(N)[:K]] = 1

        # Create matrix
        A = np.random.randn(M, N)
        A /= la.norm(A, 2)

        # Create noisy observation vector
        b = A @ x + sigma * np.random.randn(M)

        # Initial iterate
        x0 = np.zeros(N)

        return SparseLeastSquaresProblem(A, A.T, b, mu, x=x), x0

    def solve(self, x0, fasta_options=NO_ARGS):
        """Solve the L1-penalized least squares problem.

        :param x0: An initial guess for the solution
        :param fasta_options: Additional options for the FASTA algorithm (default: None)
        """
        f = lambda z: .5 * la.norm((z - self.b).ravel())**2
        gradf = lambda z: z - self.b
        g = lambda x: self.mu * la.norm(x.ravel(), 1)
        proxg = lambda x, t: proximal.shrink(x, t*self.mu)

        x = fasta(self.A, self.At, f, gradf, g, proxg, x0, **fasta_options)

        return x.solution, x

    def plot(self, solution):
        plots.plot_signals("Sparse Least Squares", self.x, solution)


if __name__ == "__main__":
    problem, x0 = SparseLeastSquaresProblem.construct()
    print("Constructed sparse least squares problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("Sparse Least Squares", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plt.show()

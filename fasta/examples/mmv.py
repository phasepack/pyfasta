"""Solve the multiple measurement vector (MMV) problem, min_X mu*MMV(X) + .5||AX-B||^2, using the FASTA solver.

X is a matrix, and so the norm ||AX-B|| is the Frobenius norm. The problem assumes that each column of X has the same
sparsity pattern, and so the sparsity constraint on X is formulated as,

    MMV(X) = sum_i ||X_i||,

where X_i denotes the ith row of X."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes, NO_ARGS

__author__ = "Noah Singer"

__all__ = ["MMVProblem"]


class MMVProblem(ExampleProblem):
    def __init__(self, A, At, B, mu, X=None):
        """Create a multiple-measurement vector problem.

        :param A: The measurement operator (must be linear, often simply a matrix)
        :param At: The Hermitian adjoint operator of A (for real matrices A, just the transpose)
        :param B: The observation matrix
        :param mu: The regularization parameter
        :param X: The problem's true solution, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.A = A
        self.At = At
        self.B = B
        self.mu = mu
        self.X = X

    @staticmethod
    def construct(M=20, N=30, L=10, K=7, sigma=0.1, mu=1.0):
        """Construct a sample max-norm problem by creating a two-moons segmentation dataset, converting it to a weighted graph, and then performing max-norm regularization on its adjacency matrix.

        :param M: The number of measurements (default: 20)
        :param N: The number of rows in the sparse matrix (default: 30)
        :param L: The number of columns in the sparse matrix (default: 10)
        :param K: The signal sparsity (default: 7)
        :param sigma: The noise level in the measurement vector (default: 0.1)
        :param mu: The regularization parameter (default: 1.0)
        """
        # Create sparse signal
        X = np.zeros((N, L))
        X[np.random.permutation(N)[:K],] = np.random.randn(K, L)

        # Create matrix
        A = np.random.randn(M, N)

        # Create noisy observation matrix
        B = A @ X + sigma * np.random.randn(M, L)

        # Initial iterate
        X0 = np.zeros((N, L))

        return MMVProblem(A, A.T, B, mu, X=X), X0

    def solve(self, X0, fasta_options=NO_ARGS):
        """Solve the multiple measurement vector (MMV) problem.

        :param X0: An initial guess for the solution
        :param fasta_options: Additional options for the FASTA algorithm (default: None)
        """
        f = lambda Z: .5 * la.norm((Z - self.B).ravel())**2
        gradf = lambda Z: Z - self.B
        g = lambda X: self.mu * np.sum(np.sqrt(np.sum(X * X, axis=1)))

        def prox_mmv(X, t):
            norms = la.norm(X, axis=1)

            # Shrink the norms, and ensure we don't divide by zero
            scale = proximal.shrink(norms, t) / (norms + (norms == 0))

            return X * scale[:,np.newaxis]

        proxg = lambda X, t: prox_mmv(X, self.mu * t)

        X = fasta(self.A, self.At, f, gradf, g, proxg, X0, **fasta_options)

        return X.solution, X

    def plot(self, solution):
        plots.plot_matrices("MMV", self.X, solution)


if __name__ == "__main__":
    problem, x0 = MMVProblem.construct()
    print("Constructed MMV problem.")

    adaptive, accelerated, plain = test_modes(problem, x0)

    plots.plot_convergence("MMV", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plt.show()

"""Solve the L1-penalized non-negative matrix factorization problem, min_{X,Y} mu||X||_1 + ||S - XY^T||, X >= 0, Y >= 0, ||Y||_inf <= 1, using the FASTA solver.

This problem is non-convex, but FBS is still often effective."""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

from fasta import fasta, proximal, plots
from fasta.examples import ExampleProblem, test_modes, NO_ARGS

__author__ = "Noah Singer"

__all__ = ["NonNegativeFactorizationProblem"]


class NNFactorizationProblem(ExampleProblem):
    def __init__(self, S, mu, X=None, Y=None):
        """Create an instance of the non-negative factorization problem.

        :param S: The matrix to factorize
        :param mu: The regularization parameter
        :param X: The matrix's true first factor, if known (default: None)
        :param Y: The matrix's true second factor, if known (default: None)
        """
        super(ExampleProblem, self).__init__()

        self.S = S
        self.mu = mu
        self.X = X
        self.Y = Y

    @staticmethod
    def construct(M=800, N=200, K=10, b=0.75, sigma=0.1, mu=1.0):
        """Construct a sample non-negative factorization problem by computing two random matrices, making one sparse, and taking their product.

        :param M: The number of rows in the data matrix (default: 800)
        :param N: The number of columns in the data matrix (default: 200)
        :param K: The rank of the factorization (default: 10)
        :param b: The sparsity parameter for the first factor, X (default: 0.75)
        :param sigma: The noise level in the observation matrix (default: 0.1)
        :param mu: The regularization parameter (default: 1.0)
        """
        # Create random factor matrices
        X = np.random.rand(M, K)
        Y = np.random.rand(N, K)

        # Make X sparse
        X *= np.random.rand(M, K) > b

        # Create observation matrix
        S = X @ Y.T + sigma * np.random.randn(M, N)

        # Initial iterates
        X0 = np.zeros((M, K))
        Y0 = np.random.rand(N, K)

        return NNFactorizationProblem(S, mu, X=X, Y=Y), (X0, Y0)

    def solve(self, inits, fasta_options=NO_ARGS):
        """Solve the L1-penalized non-negative matrix factorization problem.

        :param inits: A tuple containing the initial guesses for X0 and Y0, respectively
        :param kwargs: Additional options for the FASTA algorithm (default: None)
        """
        # Combine unknowns into single matrix so FASTA can handle them
        Z0 = np.concatenate(inits)

        # First N rows of Z are X, so X = Z[:N,...], Y = Z[N:,...]
        N = inits[0].shape[0]

        f = lambda Z: .5 * la.norm((self.S - Z[:N,...] @ Z[N:,...].T).ravel())**2

        def gradf(Z):
            # Split the iterate matrix into the X and Y matrices
            X = Z[:N,...]
            Y = Z[N:,...]

            # Compute the actual gradient
            d = X @ Y.T - self.S
            return np.concatenate((d @ Y, d.T @ X))

        g = lambda Z: self.mu * la.norm(Z[:N,...].ravel(), 1)
        proxg = lambda Z, t: np.concatenate((proximal.shrink(Z[:N,...], t*self.mu),
                                             np.minimum(np.maximum(Z[N:,...], 0), 1)))

        Z = fasta(None, None, f, gradf, g, proxg, Z0, **fasta_options)

        return (Z.solution[:N,...], Z.solution[N:,...]), Z

    def plot(self, solutions):
        # Plot the recovered matrices
        plots.plot_matrices("Factor X", self.X, solutions[0])
        plots.plot_matrices("Factor Y", self.Y, solutions[1])


if __name__ == "__main__":
    problem, inits = NNFactorizationProblem.construct()
    print("Constructed non-negative matrix factorization problem.")

    adaptive, accelerated, plain = test_modes(problem, inits)

    plots.plot_convergence("Non-Negative Matrix Factorization", (adaptive[1], accelerated[1], plain[1]), ("Adaptive", "Accelerated", "Plain"))
    problem.plot(adaptive[0])
    plt.show()

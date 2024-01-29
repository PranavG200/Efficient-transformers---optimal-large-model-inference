'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
import torch
import cupy
from cupyx.scipy.sparse.linalg import lsmr

class LinearSolver:
    """
    A class for solving linear equations using different methods.
    """

    @staticmethod
    def closed_form_solution(A, B):
        """
        Solve a linear system of equations using the closed-form solution.

        Args:
            A (torch.Tensor): Coefficient matrix.
            B (torch.Tensor): Right-hand side vector.

        Returns:
            torch.Tensor: Solution vector.
        """
        with torch.no_grad():
            if B.shape[0] == 1:
                # Handle the case of a single equation
                X = B / A[0, 0]
            else:
                # Compute the solution using matrix inversion
                X = torch.inverse(A.cpu()).to(A.device) @ B
        return X

    @staticmethod
    def lsmr_cupy_solution(A, B):
        """
        Solve a linear system of equations using LSMR with CuPy.

        Args:
            A (torch.Tensor): Coefficient matrix.
            B (torch.Tensor): Right-hand side vector.

        Returns:
            tuple: A tuple containing the solution vector and a boolean indicating convergence.
        """
        with torch.no_grad():
            # Subtract the row sums from B
            B = B - A.sum(dim=1)

            if B.shape[0] == 1:
                # Handle the case of a single equation
                X = B / A[0, 0]
                return X, True

            # Convert to CuPy arrays for LSMR
            CU_A = cupy.asarray(A.cpu().numpy())
            CU_B = cupy.asarray(B.cpu().numpy())

            # Solve using LSMR with CuPy
            solution = lsmr(CU_A, CU_B, damp=1)

            # Convert the solution back to PyTorch tensor
            X = torch.from_numpy(cupy.asnumpy(solution[0])).to(A.device)

            # Check if convergence was achieved (3 iterations)
            converged = solution[1] < 3

        return X, converged
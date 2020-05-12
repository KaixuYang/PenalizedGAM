from patsy import dmatrix
from typing import List, Union
import numpy as np
import torch
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


class bSpline:
    """
    A class to obtain b-Spline basis
    """

    def __init__(self, df: int, degree: int = 3, intercept: bool = False,
                 boundary: List[Union[float, int]] = None):
        """
        initialization function
        @param df: df of basis matrix
        @param degree: degree of polynomial, 3 indicates cubic spline
        @param intercept: whether to include intercept
        @param boundary: boundary knots, will be the min and max of x if not specified
        """
        assert df >= 0, "df must be non-negative"
        self.df = df
        self.degree = degree
        self.intercept = intercept
        self.boundary = boundary

    def basis_column_(self, x):
        """
        extract basis matrix for a column vector x
        @param x: column vector
        @return: the basis matrix
        """
        matrix = dmatrix("bs(x, df=self.df, degree=self.degree, include_intercept=self.intercept)",
                         {"train": x},
                         return_type="matrix")
        return np.array(matrix)

    def basis(self, x: Union[np.ndarray, torch.Tensor]):
        """
        extracts the basis matrix of x
        @param x: input matrix
        @return: the basis matrix
        """
        _, p = x.shape
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if p == 0:
            return None
        elif p == 1:
            return self.basis_column_(x)[:, 1:]
        else:
            basis_matrix = None
            for j in range(p):
                if basis_matrix is None:
                    basis_matrix = self.basis_column_(x[:, j])[:, 2:]
                else:
                    basis_matrix = np.concatenate([basis_matrix, self.basis_column_(x[:, j])[:, 2:]], axis=1)
            return basis_matrix

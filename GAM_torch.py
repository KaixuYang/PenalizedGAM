import numpy as np
from bSpline import bSpline
from GroupLasso import groupLasso
from sklearn.model_selection import KFold
from itertools import chain
import torch
from typing import Union, List
from utils import sigmoid, numpy_to_torch, add_intercept, remove_intercept, compute_nonzeros


class GAM(groupLasso):
    """
    This class implements the group lasso penalized GAM with B-spline basis expansion.
    """
    def __init__(self, learning_rate: float = 1, learning_rate_shrink: float = 0.5, tol: float = 1e-3,
                 data_class: str = 'regression', degree: int = 3, df: int = 5):
        """
        initialize the class
        :param learning_rate: step size of gradient descent
        :param learning_rate_shrink: shrink ratio of backtracking line search
        :param tol: tolerance
        :param data_class: 'regression', 'classification', 'gamma', 'poisson'
        :param degree: degree of B-spline, default cubic B-spline
        :param df: df of knots, evenly distributed
        :@param knot_dist: 'even' for evenly distributed knots or 'percentile' for percentile evenly distributed knots
        """
        super(GAM, self).__init__(learning_rate, learning_rate_shrink, tol, data_class)
        self.degree = degree
        self.df = df
        self.beta = None  # group lasso coefficients
        self.best_lam = None  # the best lambda of group lasso
        self.beta_gic = None  # the beta from gic
        self.beta_agl_gic = None  # the beta for agl gic
        self.intercept = None  # whether to include intercept
        self.beta_agl = None  # adaptive group lasso coefficients
        self.best_lam_agl = None  # adaptive group lasso best lambda

    @staticmethod
    def accuracy(y1: torch.Tensor, y2: torch.Tensor):
        """computes the accuracy score"""
        return 1 - torch.mean(torch.abs(y1 - y2)).item()

    @staticmethod
    def mse(y1: torch.Tensor, y2: torch.Tensor):
        """computes the mean squared error"""
        return torch.mean((y1 - y2) ** 2)

    @staticmethod
    def basis_expansion_(x: Union[np.ndarray, torch.Tensor], df: int, degree: int) -> torch.Tensor:
        """
        perform a basis expansion of the design matrix, uses B-spline with evenly distributed knots.
        :param x: the design matrix
        :param df: df of B-spline, decides the number of knots with degree
        :param degree: degree of B-spline, 3 indicates cubic B-spline
        :return: the basis matrix
        """
        basis_expansion = bSpline(df=df, degree=degree)
        basis_matrix = basis_expansion.basis(x)
        return torch.from_numpy(basis_matrix)

    def compute_weights(self, beta_hat: torch.Tensor, intercept: bool):
        """computes the adaptive group lasso weights"""
        if intercept:
            beta_hat = beta_hat[1:]
        weights = []
        for i in range(len(beta_hat) // self.df):
            weights.append(torch.norm(beta_hat[self.df * i: self.df * (i + 1)]).item())
        nonzero_idx = [i for i, j in enumerate(weights) if j != 0]
        nonzero_weights = [1 / weight for weight in weights if weight != 0]
        return nonzero_weights, nonzero_idx

    def agl_transformx(self, x: torch.Tensor, weights: List[float], weights_idx: List[int]):
        """transform matrix x to fit adaptive group lasso"""
        x_sub_idx = [list(range(i * self.df, (i + 1) * self.df)) for i in weights_idx]
        x_sub_idx = list(chain.from_iterable(x_sub_idx))
        x_sub = x[:, x_sub_idx]
        for i in range((x_sub.shape[1]) // self.df):
            x_sub[:, i * self.df: (i + 1) * self.df] /= weights[i]
        return x_sub

    def agl_transformbeta(self, beta_hat: torch.Tensor, weights: List[float], weights_idx: List[int], intercept: bool,
                          p: int):
        """transform beta back in adaptive group lasso"""
        if intercept:
            intercept_temp, beta_hat = beta_hat[0], beta_hat[1:]
        for i in range(len(beta_hat) // self.df):
            beta_hat[i * self.df: (i + 1) * self.df] /= weights[i]
        beta_full = [intercept_temp.reshape(1)] if intercept else []
        for i in range(p):
            if i in weights_idx:
                pos = weights_idx.index(i)
                beta_full.append(beta_hat[pos * self.df: (pos + 1) * self.df])
            else:
                beta_full.append(torch.zeros(self.df))
        return torch.cat(beta_full)

    def fit(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
            lam: Union[float, int], max_iters: int = 1000, intercept: bool = True):
        """fit the GAM model"""
        self.intercept = intercept
        x = remove_intercept(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        group_size = [self.df] * x.shape[1]
        self.beta = self.solve(x_basis, y, lam, group_size, max_iters, intercept)
        return self

    def fit_agl(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                lam: Union[float, int], max_iters: int = 1000, intercept: bool = True):
        """fits the adaptive group lasso"""
        x = remove_intercept(x)
        self.df += 1
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        weights, weights_idx = self.compute_weights(self.beta, intercept)
        x_sub = self.agl_transformx(x_basis, weights, weights_idx)
        group_size = [self.df] * len(weights)
        beta_agl = self.solve(x_sub, y, lam, group_size, max_iters, intercept)
        self.beta_agl = self.agl_transformbeta(beta_agl, weights, weights_idx, intercept, x.shape[1])
        return self

    def compute_gic(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, an: Union[int, float],
                    group_size: List[int]):
        """computes the generalized information criterion"""
        likelihood = self.compute_like(x, y, beta)
        num_nonzero = compute_nonzeros(beta, group_size)
        return -2 * likelihood.detach().item() + an * num_nonzero

    def fit_gic(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                lams: List[Union[float, int]], max_iters: int = 1000, intercept: bool = True,
                an: Union[int, float] = None):
        """fits the group lasso with gic"""
        self.intercept = intercept
        x = remove_intercept(x)
        self.df += 1
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        group_size = [self.df] * x.shape[1]
        gic = np.inf
        best_lam = 0
        best_beta = None
        if self.intercept:
           x_basis, group_size = add_intercept(x_basis, group_size)
        if an is None:
            an = np.log(np.log(x.shape[0])) * np.log(x.shape[1])
        for lam in lams:
            beta_gl = self.solve(x_basis, y, lam, group_size, max_iters, intercept)
            gic_temp = self.compute_gic(x_basis, y, beta_gl, an, group_size)
            if gic_temp < gic:
                gic = gic_temp
                best_lam = lam
                best_beta = beta_gl[:]
        self.beta_gic = best_beta
        self.best_lam = best_lam
        print(f"The best lam {best_lam} and the best gic {gic}.")
        return self

    def fit_agl_gic(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                    lams: List[Union[float, int]], max_iters: int = 1000, intercept: bool = True,
                    an: Union[int, float] = None):
        """fits the adaptive group lasso with gic"""
        self.intercept = intercept
        x = remove_intercept(x)
        self.df += 1
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        gic = np.inf
        best_lam = 0
        best_beta = None
        if an is None:
            an = 2 * np.log(np.log(x.shape[0])) * np.log(x.shape[1])
        weights, weights_idx = self.compute_weights(self.beta_gic, intercept)
        x_sub = self.agl_transformx(x_basis, weights, weights_idx)
        group_size = [self.df] * len(weights)
        if self.intercept:
            x_basis, group_size = add_intercept(x_sub, group_size)
        for lam in lams:
            beta_gl = self.solve(x_basis, y, lam, group_size, max_iters, intercept)
            gic_temp = self.compute_gic(x_basis, y, beta_gl, an, group_size)
            if gic_temp < gic:
                gic = gic_temp
                best_lam = lam
                best_beta = beta_gl[:]
        beta_gic = best_beta
        self.best_lam_agl = best_lam
        print(f"The best lam {best_lam} and the best gic {gic}.")
        self.beta_agl_gic = self.agl_transformbeta(beta_gic, weights, weights_idx, intercept, x.shape[1])
        return self

    def fit_cv(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], group_size: List[int],
               cv_folds: int = 5,
               lams: List[Union[float, int]] = None, max_iters: int = 1000, add_intercept: bool = True):
        """fit the GAM with cross-validation"""
        metrics = {
            'regression': self.mse,
            'classification': self.accuracy,
            'gamma': self.mse,
            'poisson': self.mse
        }
        compute_metric = getattr(metrics, self.data_class)

    def predict(self, x: Union[np.ndarray, torch.Tensor]):
        """predicts x"""
        x = numpy_to_torch(x)
        self.df += 1
        x = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        if self.intercept:
            x = add_intercept(x)
        eta = torch.matmul(x, self.beta)
        if self.data_class == 'regression':
            return eta
        elif self.data_class == 'classification':
            return torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
        elif self.data_class == 'gamma':
            return torch.exp(eta)
        else:
            return torch.round(torch.exp(eta))

    def predict_agl(self, x: Union[np.ndarray, torch.Tensor]):
        """predicts x"""
        x = numpy_to_torch(x)
        self.df += 1
        x = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        if self.intercept:
            x = add_intercept(x)
        eta = torch.matmul(x, self.beta_agl)
        if self.data_class == 'regression':
            return eta
        elif self.data_class == 'classification':
            return torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
        elif self.data_class == 'gamma':
            return torch.exp(eta)
        else:
            return torch.round(torch.exp(eta))

    def predict_gic(self, x: Union[np.ndarray, torch.Tensor]):
        """predicts x"""
        x = numpy_to_torch(x)
        self.df += 1
        x = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        if self.intercept:
            x = add_intercept(x)
        eta = torch.matmul(x, self.beta_gic)
        if self.data_class == 'regression':
            return eta
        elif self.data_class == 'classification':
            return torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
        elif self.data_class == 'gamma':
            return torch.exp(eta)
        else:
            return torch.round(torch.exp(eta))

    def predict_agl_gic(self, x: Union[np.ndarray, torch.Tensor]):
        """predicts x"""
        x = numpy_to_torch(x)
        self.df += 1
        x = self.basis_expansion_(x, self.df, self.degree)
        self.df -= 1
        if self.intercept:
            x = add_intercept(x)
        eta = torch.matmul(x, self.beta_agl_gic)
        if self.data_class == 'regression':
            return eta
        elif self.data_class == 'classification':
            return torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
        elif self.data_class == 'gamma':
            return torch.exp(eta)
        else:
            return torch.round(torch.exp(eta))
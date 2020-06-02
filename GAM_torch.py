import numpy as np
from bSpline import bSpline
from GroupLasso import groupLasso
from sklearn.model_selection import KFold
from itertools import chain
import torch
from typing import Union, List
from utils import sigmoid, numpy_to_torch, add_intercept, remove_intercept, compute_nonzeros
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


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
        self.beta_agl = None  # adaptive group lasso coefficients
        self.best_lam_agl = None  # adaptive group lasso best lambda
        self.normalize_pars = None  # normalizing columns parameters
        self.path = None  # solution path
        self.basis_expansion = None  # basis expansion info

    @staticmethod
    def accuracy(y1: torch.Tensor, y2: torch.Tensor):
        """computes the accuracy score"""
        return 1 - torch.mean(torch.abs(y1 - y2)).item()

    @staticmethod
    def mse(y1: torch.Tensor, y2: torch.Tensor):
        """computes the mean squared error"""
        return torch.mean((y1 - y2) ** 2)

    def basis_expansion_(self, x: Union[np.ndarray, torch.Tensor], df: int, degree: int) -> torch.Tensor:
        """
        perform a basis expansion of the design matrix, uses B-spline with evenly distributed knots.
        :param x: the design matrix
        :param df: df of B-spline, decides the number of knots with degree
        :param degree: degree of B-spline, 3 indicates cubic B-spline
        :param x_type: 'train' or 'test
        :return: the basis matrix
        """
        basis_expansion = bSpline(df=df, degree=degree)
        basis_matrix = basis_expansion.basis(x)
        self.basis_expansion = basis_expansion
        return torch.from_numpy(basis_matrix)

    def basis_expansion_test_(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        perform a basis expansion of the design matrix, uses B-spline with evenly distributed knots.
        :param x: the design matrix
        :param df: df of B-spline, decides the number of knots with degree
        :param degree: degree of B-spline, 3 indicates cubic B-spline
        :param x_type: 'train' or 'test
        :return: the basis matrix
        """
        basis_matrix = self.basis_expansion.basis_new(x)
        return torch.from_numpy(basis_matrix)

    def normalize(self, x: torch.Tensor):
        """normalizes x"""
        self.normalize_pars = []
        x_new = x.clone()
        for i in range(x.shape[1]):
            minimum, norm = x[:, i].min().item(), torch.norm(x[:, i]).item()
            self.normalize_pars.append([minimum, norm])
            x_new[:, i] -= minimum
            x_new[:, i] /= norm
        return x_new

    def normalize_test(self, x: torch.Tensor):
        """normalizes test data"""
        x_new = x.clone()
        for i in range(x.shape[1]):
            x_new[:, i] -= self.normalize_pars[i][0]
            x_new[:, i] /= self.normalize_pars[i][1]
        return x_new

    def compute_weights(self, beta_hat: torch.Tensor):
        """computes the adaptive group lasso weights"""
        beta_hat = beta_hat[1:]
        weights = [0]
        for i in range(len(beta_hat) // self.df):
            weight = torch.norm(beta_hat[self.df * i: self.df * (i + 1)]).item()
            if weight != 0:
                weights.append(1 / weight)
            else:
                weights.append(np.inf)
        return weights

    def fit(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
            lam: Union[float, int], max_iters: int = 1000, weight: List[Union[int, float]] = None,
            smooth: Union[float, int] = 0):
        """fit the GAM model"""
        x = remove_intercept(x)
        x = numpy_to_torch(x)
        x = self.normalize(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        group_size = [self.df] * x.shape[1]
        self.beta = self.solve(x_basis, y, lam, group_size, max_iters, weight, smooth=smooth)
        return self

    def fit_path(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], group_size: List[int],
                 num_lams: int = 50, max_iters: int = 1000, smooth: Union[int, float] = 0,
                 weights: List[Union[float, int]] = None) -> dict:
        """fits the group lasso solution path"""
        betas, lams = self.solution_path(x, y, num_lams, group_size, max_iters, smooth=smooth, weight=weights)
        res = {}
        for i, lam in enumerate(lams):
            res[lam] = betas[i]
        self.path = res
        return res

    def plot_solution_path(self):
        """plot the solution path"""
        self.plot_path(self.path)

    def compute_gic(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, an: Union[int, float],
                    group_size: List[int]):
        """computes the generalized information criterion"""
        likelihood = self.compute_like(x, y, beta)
        num_nonzero = compute_nonzeros(beta, group_size)[0]
        return -2 * likelihood.detach().item() + an * num_nonzero

    def fit_gic(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                num_lams: int, max_iters: int = 1000,
                an: Union[int, float] = None, smooth: Union[int, float] = 0):
        """fits the group lasso with gic"""
        x = numpy_to_torch(x)
        x = remove_intercept(x)
        x = self.normalize(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        group_size = [self.df] * x.shape[1]
        x_basis, group_size = add_intercept(x_basis, group_size)
        result = self.fit_path(x_basis, y, group_size, num_lams, max_iters, smooth=smooth)
        best_gic = np.inf
        if an is None:
            an = self.df * np.log(np.log(x.shape[0])) * np.log(x.shape[1]) / x.shape[0]
        for lam in result.keys():
            gic = self.compute_gic(x_basis, y, result[lam], an, group_size)
            # print(f"lam:{lam}, gic:{gic}")
            if gic < best_gic:
                best_lam = lam
                best_beta = result[lam]
                best_gic = gic
        self.beta_gic = best_beta
        self.beta = best_beta
        print(f"The best lam {best_lam} and the best gic {best_gic}.")
        return self

    def fit_agl(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                lam: Union[float, int], max_iters: int = 1000,
                smooth: Union[float, int] = 0, weights: List[Union[int, float]] = None):
        """fits the adaptive group lasso"""
        if self.beta is None and weights is None:
            print("Initial beta estimation is not available, please run fit or fit_gic first.")
            return None
        if weights is None:
            weights = self.compute_weights(self.beta)
        x = remove_intercept(x)
        x = numpy_to_torch(x)
        x = self.normalize(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        group_size = [self.df] * len(weights)
        x_basis, group_size = add_intercept(x_basis, group_size)
        beta_agl = self.solve(x_basis, y, lam, group_size, max_iters, weights, smooth=smooth)
        self.beta_agl = beta_agl
        self.beta = beta_agl
        return self

    def fit_agl_gic(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                    num_lams: int, max_iters: int = 1000,
                    an: Union[int, float] = None, smooth: Union[float, int] = 0):
        """fits the adaptive group lasso with gic"""
        weights = self.compute_weights(self.beta_gic)
        x = numpy_to_torch(x)
        x = remove_intercept(x)
        x = self.normalize(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        group_size = [self.df] * x.shape[1]
        x_basis, group_size = add_intercept(x_basis, group_size)
        result = self.fit_path(x_basis, y, group_size, num_lams, max_iters, smooth=smooth, weights=weights)
        best_gic = np.inf
        best_lam = 0
        best_beta = None
        if an is None:
            an = np.log(np.log(x.shape[0])) * np.log(x.shape[1]) / x.shape[0]
        for lam in result.keys():
            beta_full = result[lam]
            gic = self.compute_gic(x_basis, y, beta_full, an, group_size)
            print(f"lam:{lam}, gic:{gic}")
            if gic < best_gic:
                best_lam = lam
                best_beta = beta_full
                best_gic = gic
        self.beta_agl_gic = best_beta
        self.beta = best_beta
        num_nz, nz = compute_nonzeros(best_beta, group_size)
        print(f"The best lam {best_lam} and the best gic {best_gic}. Finally selected {num_nz - 1} nonzeros: {nz}")
        return self

    def fit_2(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
              num_lams: int, max_iters: int = 1000,
              an: Union[int, float] = None, smooth: Union[float, int] = 0):
        """fit group lasso then followed by adaptive group lasso, saves time for basis expansion"""
        x = numpy_to_torch(x)
        x = remove_intercept(x)
        x = self.normalize(x)
        x_basis = self.basis_expansion_(x, self.df, self.degree)
        group_size = [self.df] * x.shape[1]
        x_basis, group_size = add_intercept(x_basis, group_size)
        result = self.fit_path(x_basis, y, group_size, num_lams, max_iters, smooth=smooth)
        beta_gl = result[min(list(result.keys()))]
        weights = self.compute_weights(beta_gl)
        result = self.fit_path(x_basis, y, group_size, num_lams, max_iters, smooth=smooth, weights=weights)
        best_gic = np.inf
        best_lam = 0
        best_beta = None
        if an is None:
            an = np.log(np.log(x.shape[0])) * np.log(x.shape[1]) / x.shape[0]
        for lam in result.keys():
            beta_full = result[lam]
            gic = self.compute_gic(x_basis, y, beta_full, an, group_size)
            print(f"lam:{lam}, gic:{gic}")
            if gic < best_gic:
                best_lam = lam
                best_beta = beta_full
                best_gic = gic
        self.beta_agl_gic = best_beta
        self.beta = best_beta
        num_nz, nz = compute_nonzeros(best_beta, group_size)
        print(f"The best lam {best_lam} and the best gic {best_gic}. Finally selected {num_nz - 1} nonzeros: {nz}")
        return self

    def predict(self, x: Union[np.ndarray, torch.Tensor]):
        """predicts x"""
        x = numpy_to_torch(x)
        x = remove_intercept(x)
        x = self.normalize_test(x)
        x_basis = self.basis_expansion_test_(x)
        x_basis = add_intercept(x_basis)
        eta = torch.matmul(x_basis, self.beta)
        if self.data_class == 'regression':
            return eta
        elif self.data_class == 'classification':
            return torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
        elif self.data_class == 'gamma':
            return torch.exp(-eta)
        else:
            return torch.round(torch.exp(eta))

    def predict_mse(self, x_test: torch.Tensor, y_test: torch.Tensor):
        """predicts the path"""
        x = numpy_to_torch(x_test)
        x = self.normalize_test(x)
        x = self.basis_expansion_test_(x)
        x = add_intercept(x)
        mses = []
        for lam, beta in self.path.items():
            eta = torch.matmul(x, beta)
            if self.data_class == 'regression':
                y = eta
            elif self.data_class == 'classification':
                y = torch.where(sigmoid(eta) > 0.5, torch.ones(len(eta)), torch.zeros(len(eta)))
            elif self.data_class == 'gamma':
                y = torch.exp(-eta)
            else:
                y = torch.round(torch.exp(eta))
            mses.append(torch.mean((y - y_test) ** 2).item())
            print(f"lam is {lam}, mse is {mses[-1]}, "
                  f"number of nonzeros is {compute_nonzeros(beta, [1] + [self.df] * x.shape[1])[0] - 1}")




    def fit_cv(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], group_size: List[int],
               cv_folds: int = 5,
               lams: List[Union[float, int]] = None, max_iters: int = 1000, add_intercept: bool = True,
               smooth: Union[float, int] = None):
        """fit the GAM with cross-validation"""
        metrics = {
            'regression': self.mse,
            'classification': self.accuracy,
            'gamma': self.mse,
            'poisson': self.mse
        }
        compute_metric = getattr(metrics, self.data_class)



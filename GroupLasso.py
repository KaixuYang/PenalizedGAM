import numpy as np
import torch
from warnings import warn
from typing import List, Union
from utils import check_xy, sigmoid, numpy_to_torch, add_intercept
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

class groupLasso:
    """
    This class solves a group lasso problem
    """

    def __init__(self, learning_rate: float = 1, learning_rate_shrink: float = 0.5, tol: float = 1e-3,
                 data_class: str = 'regression'):
        """
        initialize the class
        :param learning_rate: step size of gradient descent
        :param learning_rate_shrink: shrink ratio of backtracking line search
        :param tol: tolerance
        :param data_class: 'regression', 'classification', 'gamma', 'poisson'
        """
        self.learning_rate = learning_rate
        self.learning_rate_shrink = learning_rate_shrink
        self.tol = tol
        if data_class in ['regression', 'classification', 'gamma', 'poisson']:
            self.data_class = data_class
        else:
            raise ValueError("data_class must be regression or classification.")
        self.beta = None  # group lasso coefficients
        self.R = None  # right matrices of QR decomposition of basis matrix, used to do orthogonalization.

    @staticmethod
    def group_orthogonalization(x: torch.Tensor, group_size: List[int]):
        """perform group orthogonalization"""
        right_matrices = []
        n = x.shape[0]
        g = 0
        num_variables = sum(group_size)
        start = 0
        new_x = x.clone()
        while start < num_variables:
            current_size = group_size[g]
            current_x = x[:, start: start + current_size]
            q, r = torch.qr(current_x)
            new_x[:, start: start + current_size] = q * np.sqrt(n)
            right_matrices.append(r / np.sqrt(n))
            g += 1
            start += current_size
        return new_x, right_matrices

    @staticmethod
    def group_orthogonalization_inverse(beta: torch.Tensor, right_matrices: List[torch.Tensor],
                                        group_size: List[int]):
        """transform coefficients back"""
        g = 0
        num_variables = sum(group_size)
        start = 0
        while start < num_variables:
            current_size = group_size[g]
            beta[start: start + current_size] = torch.matmul(
                torch.inverse(right_matrices[g]), beta[start: start + current_size])
            g += 1
            start += current_size
        return beta

    @staticmethod
    def initialize(group_size: List[int]):
        """initialize parameters"""
        beta = torch.zeros(sum(group_size), requires_grad=True)
        error = np.inf
        iters = 0
        loss = np.inf
        return beta, error, iters, loss

    @staticmethod
    def compute_regression_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression likelihood"""
        eta = torch.matmul(x, beta)
        like = -torch.sum((y - eta) ** 2) / x.shape[0]
        return like

    @staticmethod
    def compute_regression_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression gradient"""
        eta = torch.matmul(x, beta)
        return 2 * torch.matmul(x.t(), y - eta) / x.shape[0]

    @staticmethod
    def compute_regression_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression hessian matrix diagonal elements"""
        return -2 * torch.norm(x, 2, dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_logistic_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the logistic loss"""
        eta = torch.matmul(x, beta)
        like = torch.sum(y * eta - torch.log(1 + torch.exp(eta))) / x.shape[0]
        return like

    @staticmethod
    def compute_logistic_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression gradient"""
        eta = torch.matmul(x, beta)
        mu = sigmoid(eta)
        return torch.matmul(x.t(), y - mu) / x.shape[0]

    @staticmethod
    def compute_logistic_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression hessian matrix diagonal elements"""
        eta = torch.matmul(x, beta)
        sigma = sigmoid(eta) / (1 + torch.exp(eta))
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(torch.matmul(sigma_sqrt, x), dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_poisson_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson loss"""
        eta = torch.matmul(x, beta)
        like = torch.sum(y * eta - torch.exp(eta)) / x.shape[0]
        return like

    @staticmethod
    def compute_poisson_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson gradient"""
        eta = torch.matmul(x, beta)
        mu = torch.exp(eta)
        return torch.matmul(x.t(), y - mu) / x.shape[0]

    @staticmethod
    def compute_poisson_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson hessian matrix diagonal elements"""
        eta = torch.matmul(x, beta)
        sigma = torch.exp(eta)
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(torch.matmul(sigma_sqrt, x), dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_gamma_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the gamma loss, assume dispersion parameter nv is normalized to 1"""
        eta = torch.matmul(x, beta)
        like = torch.sum(eta - y * torch.exp(eta)) / x.shape[0]
        return like

    @staticmethod
    def compute_gamma_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson gradient"""
        eta = torch.matmul(x, beta)
        return torch.matmul(x.t(), torch.ones(x.shape[0]) - y * torch.exp(eta)) / x.shape[0]

    @staticmethod
    def compute_gamma_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson hessian matrix diagonal elements"""
        eta = torch.matmul(x, beta)
        sigma = y * torch.exp(eta)
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(torch.matmul(sigma_sqrt, x), dim=0) ** 2 / x.shape[0]

    def compute_like(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """ compute the appropriate likelihood based on the values and the regression class"""
        class_like = {
            'regression': self.compute_regression_like,
            'classification': self.compute_logistic_like,
            'poisson': self.compute_poisson_like,
            'gamma': self.compute_gamma_like
        }
        like_func = class_like.get(self.data_class)
        return like_func(x, y, beta)

    def compute_grad(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """ compute the appropriate gradient based on the values and the regression class"""
        class_grad = {
            'regression': self.compute_regression_grad,
            'classification': self.compute_logistic_grad,
            'poisson': self.compute_poisson_grad,
            'gamma': self.compute_gamma_grad
        }
        like_func = class_grad.get(self.data_class)
        return like_func(x, y, beta)

    def compute_hes(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """ compute the appropriate hessian matrix diagonal elements based on the values and the regression class"""
        class_hes = {
            'regression': self.compute_regression_hes,
            'classification': self.compute_logistic_hes,
            'poisson': self.compute_poisson_hes,
            'gamma': self.compute_gamma_hes
        }
        like_func = class_hes.get(self.data_class)
        return like_func(x, y, beta)

    @staticmethod
    def find_group_index(group_size: List[int], g: int):
        """finds the start and end index for the current group"""
        group_idx_start = sum(group_size[:g])
        group_idx_end = sum(group_size[:(g + 1)])
        return group_idx_start, group_idx_end

    def compute_hg(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, group_idx_start: int,
                    group_idx_end: int):
        """computes the approximated hessian matrix"""
        c_star = 0.1
        hessian_diag = self.compute_hes(x, y, beta)
        hg = -torch.max(hessian_diag[group_idx_start: group_idx_end]).item()
        hg = max(hg, -c_star)
        return hg

    def check_threshold(self, derivative: torch.Tensor, hg: float, beta: torch.Tensor, lam: Union[float, int],
                        group_idx_start: int, group_idx_end: int):
        """checks if the current group should be dropped"""
        diff = derivative[group_idx_start: group_idx_end] - hg * beta[group_idx_start: group_idx_end]
        group_size = group_idx_end - group_idx_start
        if torch.norm(diff, 2).item() <= lam * np.sqrt(group_size):
            return True
        else:
            return False

    def compute_d(self, set_zero: bool, derivative: torch.Tensor, beta: torch.Tensor, lam: float,
                  group_idx_start: int, group_idx_end: int, hg: float):
        """compute the gradient"""
        if set_zero:
            d_full = torch.zeros_like(beta)
            d_full[group_idx_start: group_idx_end] = -beta[group_idx_start: group_idx_end]
            return d_full
        else:
            diff = derivative[group_idx_start: group_idx_end] - hg * beta[group_idx_start: group_idx_end]
            group_size = group_idx_end - group_idx_start
            d = -(derivative[group_idx_start: group_idx_end] - lam * np.sqrt(group_size) * diff / torch.norm(diff, 2)) \
                / hg
            d_full = torch.zeros_like(beta)
            d_full[group_idx_start: group_idx_end] = d
            return -d_full

    def compute_penalized_loss(self, like: torch.Tensor, beta: torch.Tensor, group_size: List[int],
                               lam: Union[float, int]):
        """computes the penalized loss function"""
        penalty = torch.tensor(0.)
        for g in range(len(group_size)):
            start, end = self.find_group_index(group_size, g)
            penalty += lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end], 2)
        return -like + penalty

    def line_search(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, d: torch.Tensor,
                    group_size: List[int], g: int, lam: Union[float, int]):
        """perform a line search for the step size"""
        alpha = self.learning_rate
        delta = self.learning_rate_shrink
        sigma = 0.1
        start, end = self.find_group_index(group_size, g)
        like_old = self.compute_like(x, y, beta)
        loss_old = self.compute_penalized_loss(like_old, beta, group_size, lam)
        like_new = self.compute_like(x, y, beta + alpha * d)
        loss_new = self.compute_penalized_loss(like_new, beta + alpha * d, group_size, lam)
        decrease = loss_new - loss_old
        gradient = self.compute_grad(x, y, beta)
        expected_decrease = -torch.matmul(d.t(), gradient) + \
                            lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end] + d[start: end], 2) - \
                            lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end], 2)
        while decrease > alpha * sigma * expected_decrease:
            alpha *= delta
            like_new = self.compute_like(x, y, beta + alpha * d)
            loss_new = self.compute_penalized_loss(like_new, beta + alpha * d, group_size, lam)
            decrease = loss_new - loss_old
        return alpha

    def solve(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], lam: Union[float, int],
              group_size: Union[int, List[int]], max_iters: int = 1000, intercept: bool = True) \
            -> torch.Tensor:
        """
        fits the model with a use specified lambda
        :param x: the design matrix
        :param y: the response
        :param lam: the lambda for group lasso
        :param group_size: list of group sizes, or simple group size if all groups are of the same size
        :param max_iters: the maximum number of iterations
        :param intercept: whether to add intercept
        :return: coefficients
        """
        if isinstance(group_size, int):
            group_size = [group_size] * (x.shape[1] // group_size)
        assert np.sum(group_size) == x.shape[1], "Sum of group sizes do not match number of variables."
        assert lam >= 0, "Tuning parameter lam must be non-negative."
        """initialize parameters"""
        x = numpy_to_torch(x)
        y = numpy_to_torch(y)
        x, y = check_xy(x, y)
        if intercept:
            x, group_size = add_intercept(x, group_size)
        x1, self.R = self.group_orthogonalization(x, group_size)
        beta, error, iters, loss = self.initialize(group_size)
        if intercept:
            intercept_err = np.inf
        else:
            intercept_err = 0
        beta_old = beta[:]
        num_groups = len(group_size)
        """start iterations"""
        while error > self.tol and intercept_err > self.tol and iters <= max_iters:
            iters += 1
            for g in range(num_groups):
                group_idx_start, group_idx_end = self.find_group_index(group_size, g)
                hg = self.compute_hg(x1, y, beta, group_idx_start, group_idx_end)
                derivative = self.compute_grad(x1, y, beta)
                if intercept and g == 0:
                    set_zero = False
                else:
                    set_zero = self.check_threshold(derivative, hg, beta, lam, group_idx_start, group_idx_end)
                d = self.compute_d(set_zero, derivative, beta, lam, group_idx_start, group_idx_end, hg)
                alpha = self.line_search(x1, y, beta, d, group_size, g, lam)
                beta = beta + alpha * d
            if intercept:
                error = torch.norm(beta[1:] - beta_old[1:])
                intercept_err = abs(beta[0].detach().numpy() - beta_old[0].detach().numpy())
            beta_old = beta
            # print(f"error is {error}")
        beta = self.group_orthogonalization_inverse(beta, self.R, group_size)
        return beta

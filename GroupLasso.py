import numpy as np
import torch
from warnings import warn
from typing import List, Union
from utils import check_xy, sigmoid, numpy_to_torch, add_intercept, compute_nonzeros, group_norm
import os
from datetime import datetime
import matplotlib.pyplot as plt

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
        self.smoothness_penalty = None  # smoothness penalty
        self.group_size = None

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
            new_x[:, start: start + current_size] = q / np.sqrt(n)
            right_matrices.append(r * np.sqrt(n))
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
        beta_new = beta.clone()
        while start < num_variables:
            current_size = group_size[g]
            beta_new[start: start + current_size] = right_matrices[g].inverse().matmul(
                beta[start: start + current_size])
            g += 1
            start += current_size
        return beta_new

    @staticmethod
    def initialize(group_size: List[int]):
        """initialize parameters"""
        beta = torch.zeros(sum(group_size))
        error = np.inf
        iters = 0
        loss = np.inf
        return beta, error, iters, loss

    @staticmethod
    def compute_regression_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression likelihood"""
        eta = x.matmul(beta)
        like = -torch.sum((y - eta) ** 2) / x.shape[0]
        return like

    @staticmethod
    def compute_regression_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression gradient"""
        eta = x.matmul(beta)
        return 2 * x.t().matmul(y - eta) / x.shape[0]

    @staticmethod
    def compute_regression_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression hessian matrix diagonal elements"""
        return -2 * torch.norm(x, 2, dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_logistic_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the logistic loss"""
        eta = x.matmul(beta)
        like = torch.sum(y * eta - torch.log(1 + torch.exp(eta))) / x.shape[0]
        return like

    @staticmethod
    def compute_logistic_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression gradient"""
        eta = x.matmul(beta)
        mu = sigmoid(eta)
        return x.t().matmul(y - mu) / x.shape[0]

    @staticmethod
    def compute_logistic_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the regression hessian matrix diagonal elements"""
        eta = x.matmul(beta)
        sigma = sigmoid(eta) / (1 + torch.exp(eta)) ** 2
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(sigma_sqrt.matmul(x), dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_poisson_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson loss"""
        eta = x.matmul(beta)
        like = torch.sum(y * eta - torch.exp(eta)) / x.shape[0]
        return like

    @staticmethod
    def compute_poisson_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson gradient"""
        eta = x.matmul(beta)
        mu = torch.exp(eta)
        return x.t().matmul(y - mu) / x.shape[0]

    @staticmethod
    def compute_poisson_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson hessian matrix diagonal elements"""
        eta = x.matmul(beta)
        sigma = torch.exp(eta)
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(sigma_sqrt.matmul(x), dim=0) ** 2 / x.shape[0]

    @staticmethod
    def compute_gamma_like(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the gamma loss, assume dispersion parameter nv is normalized to 1"""
        eta = x.matmul(beta)
        like = torch.sum(eta - y * torch.exp(eta)) / x.shape[0]
        return like

    @staticmethod
    def compute_gamma_grad(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson gradient"""
        eta = x.matmul(beta)
        return x.t().matmul(torch.ones(x.shape[0]) - y * torch.exp(eta)) / x.shape[0]

    @staticmethod
    def compute_gamma_hes(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
        """computes the poisson hessian matrix diagonal elements"""
        eta = x.matmul(beta)
        sigma = y * torch.exp(eta)
        sigma_sqrt = torch.sqrt(torch.diag(sigma.squeeze()))
        return -torch.norm(sigma_sqrt.matmul(x), dim=0) ** 2 / x.shape[0]

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
    def generate_smooth_matrix(size: int):
        """generate the P-spline smoothness penalty matrix"""
        if size == 1:
            return 0
        mat = torch.from_numpy(np.diff(np.eye(size, size)))
        return mat.matmul(mat.t()).float()

    def compute_smoothness(self, beta: torch.Tensor, group_size: List[int]):
        """computes beta^T S beta group-wisely"""
        smo = 0
        start = 0
        last_size = -1
        for i in group_size:
            if i == 1:
                start += 1
                last_size = i
                continue
            else:
                beta_i = beta[start: start + i]
                if i != last_size:
                    mat = self.generate_smooth_matrix(i)
                smo += beta_i.t().matmul(mat).matmul(beta_i)
                start += i
                last_size = i
        return self.smoothness_penalty * smo

    def compute_smoothness_grad(self, beta: torch.Tensor, group_size: List[int]):
        """computes beta^T S beta group-wisely gradient"""
        smo_grad = torch.zeros_like(beta)
        start = 0
        last_size = -1
        for i in group_size:
            if i == 1:
                smo_grad[start: start + i] = 0
                start += 1
                last_size = i
                continue
            else:
                beta_i = beta[start: start + i]
                if i != last_size:
                    mat = self.generate_smooth_matrix(i)
                smo_grad[start: start + i] = 2 * mat.matmul(beta_i)
                start += i
                last_size = i
        return self.smoothness_penalty * smo_grad

    @staticmethod
    def find_group_index(group_size: List[int], g: int):
        """finds the start and end index for the current group"""
        group_idx_start = sum(group_size[:g])
        group_idx_end = sum(group_size[:(g + 1)])
        return group_idx_start, group_idx_end

    def compute_hg(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, group_idx_start: int,
                   group_idx_end: int, lam_smo: float = 0):
        """computes the approximated hessian matrix"""
        c_star = 0.1
        hessian_diag = self.compute_hes(x, y, beta)
        # if lam_smo is not None and lam_smo > 0:
        #     smo_diag = torch.zeros_like(hessian_diag)
        #     smo_diag[:] = 2
        #     smo_diag[0] = 1
        #     hessian_diag -= lam_smo * smo_diag
        hg = -torch.max(hessian_diag[group_idx_start: group_idx_end]).item()
        hg = max(hg, c_star)
        return hg

    @staticmethod
    def check_threshold(derivative: torch.Tensor, hg: float, beta: torch.Tensor, lam: Union[float, int],
                        group_idx_start: int, group_idx_end: int):
        """checks if the current group should be dropped"""
        diff = derivative[group_idx_start: group_idx_end] - hg * beta[group_idx_start: group_idx_end]
        group_size = group_idx_end - group_idx_start
        if torch.norm(diff, 2).item() <= lam * np.sqrt(group_size):
            return True
        else:
            return False

    @staticmethod
    def compute_d(set_zero: bool, derivative: torch.Tensor, beta: torch.Tensor, lam: float,
                  group_idx_start: int, group_idx_end: int, hg: float):
        """compute the gradient"""
        if set_zero:
            d_full = torch.zeros_like(beta)
            d_full[group_idx_start: group_idx_end] = -beta[group_idx_start: group_idx_end]
            return d_full
        else:
            diff = derivative[group_idx_start: group_idx_end] - hg * beta[group_idx_start: group_idx_end]
            group_size = group_idx_end - group_idx_start
            if group_size == 1:
                d = -derivative[group_idx_start: group_idx_end] / hg
            else:
                d = -(derivative[group_idx_start: group_idx_end]
                      - lam * np.sqrt(group_size) * diff / torch.norm(diff, 2)) / hg
            d_full = torch.zeros_like(beta)
            d_full[group_idx_start: group_idx_end] = d
            return -d_full

    def compute_penalized_loss(self, like: torch.Tensor, beta: torch.Tensor, group_size: List[int],
                               lam: Union[float, int]):
        """computes the penalized loss function"""
        penalty = torch.tensor(0.)
        for g in range(1, len(group_size)):
            start, end = self.find_group_index(group_size, g)
            penalty += lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end], 2)
        return -like + penalty

    def close_form_QM(self, beta: torch.Tensor, derivative: torch.Tensor, hg: float, lam: Union[int, float],
                      group_idx_start: int, group_idx_end: int, weight: Union[int, float],
                      smooth: Union[int, float] = 0):
        """find the closed form solution for a group using the QM method"""
        u_beta = derivative[group_idx_start: group_idx_end] + hg * beta[group_idx_start: group_idx_end]
        thres_func = (1 - lam * weight / torch.norm(u_beta)).item()
        if thres_func < 0:
            return torch.zeros(group_idx_end - group_idx_start)
        else:
            if smooth > 0:
                s = self.generate_smooth_matrix(group_idx_end - group_idx_start)
                u_norm = torch.norm(u_beta)
                multiplier = (u_norm / (u_norm - lam * weight)) * torch.eye(group_idx_end - group_idx_start) \
                             + smooth * s
                return multiplier.inverse().matmul(u_beta) / hg
            else:
                return u_beta * thres_func / hg

    def line_search(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, d: torch.Tensor,
                    group_size: List[int], g: int, lam: Union[float, int]):
        """perform a line search for the step size"""
        alpha = self.learning_rate
        delta = self.learning_rate_shrink
        sigma = 0.1
        start, end = self.find_group_index(group_size, g)
        like_old = self.compute_like(x, y, beta)
        loss_old = self.compute_penalized_loss(like_old, beta, group_size, lam)
        new_beta = beta + alpha * d
        like_new = self.compute_like(x, y, new_beta)
        loss_new = self.compute_penalized_loss(like_new, new_beta, group_size, lam)
        if self.smoothness_penalty is not None:
            loss_old += self.compute_smoothness(beta, group_size)
            loss_new += self.compute_smoothness(new_beta, group_size)
        decrease = loss_new - loss_old
        gradient = self.compute_grad(x, y, beta)
        if self.smoothness_penalty is not None:
            gradient -= self.compute_smoothness_grad(beta, group_size)
        expected_decrease = -torch.matmul(d.t(), gradient)
            # lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end] + d[start: end], 2) - \
            # lam * np.sqrt(group_size[g]) * torch.norm(beta[start: end], 2)
        while decrease > alpha * sigma * expected_decrease:
            alpha *= delta
            new_beta = beta + alpha * d
            like_new = self.compute_like(x, y, new_beta)
            loss_new = self.compute_penalized_loss(like_new, new_beta, group_size, lam)
            if self.smoothness_penalty is not None:
                loss_new += self.compute_smoothness(new_beta, group_size)
            decrease = loss_new - loss_old
        return alpha

    def null_estimate(self, y: torch.Tensor):
        """return the null model intercept"""
        mean = y.mean().item()
        if self.data_class == 'regression':
            return mean
        elif self.data_class == 'classification':
            return np.log(mean / (1 - mean))
        elif self.data_class == 'gamma':
            return -np.log(mean)
        else:
            return np.log(mean)

    def find_max_lambda(self, x: torch.Tensor, y: torch.Tensor, weights: List[float], group_size: List[int]):
        """find the smallest lambda that corresponds to no active variables"""
        beta = torch.tensor([self.null_estimate(y)] + [0] * sum(group_size))
        grad = self.compute_grad(x, y, beta)
        group_norms = group_norm(grad[1:], group_size)
        weights = [i if i != 0 else np.inf for i in weights]
        group_norms /= torch.tensor(weights)
        return max(group_norms)

    def solve(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], lam: Union[float, int],
              group_size: Union[int, List[int]], max_iters: int = 1000, weight: List[Union[int, List[int]]] = None,
              smooth: Union[float, int] = 0, recompute_hg: bool = True,
              beta_warm: torch.Tensor = None, weight_multiplied: bool = False) -> torch.Tensor:
        """
        fits the model with a use specified lambda
        :param x: the design matrix
        :param y: the response
        :param lam: the lambda for group lasso
        :param group_size: list of group sizes, or simple group size if all groups are of the same size
        :param weight: feature weights
        :param max_iters: the maximum number of iterations
        :param smooth: smoothness parameter
        :param recompute_hg: whether to recompute hg
        :param beta_warm: warm start of beta
        :return: coefficients
        """
        if isinstance(group_size, int):
            group_size = [group_size] * (x.shape[1] // group_size)
        assert np.sum(group_size) == x.shape[1], \
            f"Sum of group sizes {sum(group_size)} do not match number of variables {x.shape[1]}."
        assert lam >= 0, "Tuning parameter lam must be non-negative."
        """initialize parameters"""
        self.smoothness_penalty = smooth
        x = numpy_to_torch(x)
        y = numpy_to_torch(y)
        x, y = check_xy(x, y)
        x, group_size = add_intercept(x, group_size)
        if weight is None:
            weight = [1] * len(group_size)
        if not weight_multiplied:
            weights = [np.sqrt(group_size[i]) * weight[i] for i in range(len(group_size))]
        else:
            weights = weight[:]
        x1 = x.clone()
        # x1, self.R = self.group_orthogonalization(x, group_size)
        beta, error, iters, loss = self.initialize(group_size)
        if beta_warm is not None and beta_warm.shape == beta.shape:
            beta = beta_warm
        intercept_err = np.inf
        beta_old = beta.clone()
        num_groups = len(group_size)
        hg = None
        """start iterations"""
        while (error > self.tol or intercept_err > self.tol) and iters <= max_iters:
            iters += 1
            for g in range(num_groups):
                group_idx_start, group_idx_end = self.find_group_index(group_size, g)
                if recompute_hg or hg is None or g <= 2:
                    hg = self.compute_hg(x1, y, beta, group_idx_start, group_idx_end)
                derivative = self.compute_grad(x1, y, beta)
                # if self.smoothness_penalty > 0:
                #     derivative -= self.compute_smoothness_grad(beta, group_size)
                if g == 0:
                    d = self.compute_d(False, derivative, beta, lam, group_idx_start, group_idx_end, hg)
                    alpha = self.line_search(x1, y, beta, d, group_size, g, lam)
                    beta = beta + alpha * d
                else:
                    beta[group_idx_start: group_idx_end] = self.close_form_QM(beta, derivative, hg, lam,
                                                                              group_idx_start, group_idx_end,
                                                                              weights[g], smooth)
            error = torch.norm(beta[1:] - beta_old[1:])
            intercept_err = abs(beta[0].detach().numpy() - beta_old[0].detach().numpy())
            beta_old = beta.clone()
            # print(f"error is {error}")
        # beta = self.group_orthogonalization_inverse(beta, self.R, group_size)
        return beta

    def strong_rule(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, group_size: List[int],
                    lam: Union[int, float], lam_last: Union[int, float], weight: List[Union[int, List[int]]]) \
            -> List[int]:
        """filter the groups that satisfy the strong rule"""
        x = numpy_to_torch(x)
        y = numpy_to_torch(y)
        derivative = self.compute_grad(x, y, beta)
        strong_index = []
        for g in range(len(group_size)):
            group_idx_start, group_idx_end = self.find_group_index(group_size, g)
            if group_idx_end - group_idx_start == 1:
                strong_index.append(g)
            elif weight[g] == np.inf:
                continue
            else:
                left = torch.norm(-derivative[group_idx_start: group_idx_end])
                if 2 * lam - lam_last == 0:
                    right = np.inf
                else:
                    right = weight[g] * (2 * lam - lam_last)
                if left >= right:
                    strong_index.append(g)
        return strong_index

    @staticmethod
    def strong_x(x: torch.Tensor, group_size: List[int], strong_index: List[int], weight: List[Union[int, List[int]]]):
        """find the sub-matrix of x that satisfies the strong rule, their group sizes and weights"""
        group_idx = []
        group_size_new = []
        weight_new = []
        start = 0
        for i in range(len(group_size)):
            if i in strong_index:
                group_idx += list(range(start, start + group_size[i]))
                group_size_new.append(group_size[i])
                weight_new.append(weight[i])
            start += group_size[i]
        return x[:, group_idx], group_size_new, weight_new

    @staticmethod
    def strong_to_full_beta(beta_s: torch.Tensor, group_size: List[int], group_index: List[int]):
        """transform the beta on S to full beta with other entries being zero"""
        beta_new = torch.zeros(sum(group_size))
        start = 0
        beta_start = 0
        for i in range(len(group_size)):
            if i in group_index:
                beta_new[start: start + group_size[i]] = beta_s[beta_start: beta_start + group_size[i]]
                beta_start += group_size[i]
            start += group_size[i]
        return beta_new

    def fail_kkt(self, x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, group_size: List[int],
                 lam: Union[int, float], group_index: List[int], weight: List[Union[int, List[int]]]) -> List[int]:
        """finds the index not in S that fails the KKT condition"""
        v = []
        for g in range(len(group_size)):
            if g in group_index or weight[g] == np.inf:
                continue
            start, end = self.find_group_index(group_size, g)
            derivative = self.compute_grad(x, y, beta)
            left = torch.norm(derivative[start: end])
            right = lam * weight[g]
            if left > right:
                v.append(g)
        return v

    def solution_path(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                      num_lams: int, group_size: Union[int, List[int]], max_iters: int = 1000,
                      smooth: Union[float, int] = 0, recompute_hg: bool = True,
                      weight: List[Union[int, List[int]]] = None) \
            -> (List[torch.Tensor], List[float]):
        """
        fits the model with a use specified lambda
        :param x: the design matrix
        :param y: the response
        :param num_lams: number of lambdas
        :param group_size: list of group sizes, or simple group size if all groups are of the same size
        :param max_iters: the maximum number of iterations
        :param smooth: smoothness parameter
        :param recompute_hg: whether to recompute hg
        :param weight: feature weights
        :return: coefficients
        """
        self.group_size = group_size
        if isinstance(group_size, int):
            group_size = [1] + [group_size] * (x.shape[1] // group_size)
        if weight is None:
            weight = [0] + [1] * len(group_size)
        weights = [np.sqrt(group_size[i]) * weight[i] for i in range(len(group_size))]
        assert np.sum(group_size) == x.shape[1], "Sum of group sizes do not match number of variables."
        betas = []
        lam_max = self.find_max_lambda(x, y, weights[1:], group_size[1:])
        lam_max *= (1 + 1 / num_lams * 10)
        lams = list(np.linspace(0, lam_max, num_lams))
        lams.remove(0)
        lams.sort(reverse=True)
        lam_last = None
        for lam in lams:
            if not betas:
                # beta_full = self.solve(x, y, lam, group_size, max_iters, weights, smooth, recompute_hg)
                beta_full = torch.tensor([self.null_estimate(y)] + [0] * (sum(group_size) - 1))
                betas.append(beta_full)
                lam_last = lam
            else:
                beta = betas[-1]
                strong_index = self.strong_rule(x, y, beta, group_size, lam, lam_last, weights)
                x_s, group_size_s, weight_s = self.strong_x(x, group_size, strong_index, weights)
                beta_s = self.solve(x_s, y, lam, group_size_s, max_iters, weight_s, smooth, recompute_hg,
                                    weight_multiplied=True)
                beta_full = self.strong_to_full_beta(beta_s, group_size, strong_index)
                v = self.fail_kkt(x, y, beta_full, group_size, lam, strong_index, weights)
                while len(v) > 0:
                    strong_index = list(set(strong_index + v))
                    x_s, group_size_s, weight_s = self.strong_x(x, group_size, strong_index, weights)
                    beta_s = self.solve(x_s, y, lam, group_size_s, max_iters, weight_s, smooth,
                                        recompute_hg, weight_multiplied=True)
                    beta_full = self.strong_to_full_beta(beta_s, group_size, strong_index)
                    v = self.fail_kkt(x, y, beta_full, group_size, lam, strong_index, weights)
                betas.append(beta_full)
                lam_last = lam
            num_nz, nz = compute_nonzeros(beta_full, group_size)
            print(f"Fitted lam = {lam}, {num_nz - 1} nonzero variables {nz}")
            if sum([group_size[i] for i in nz]) > x.shape[0]:
                lams = lams[:lams.index(lam) + 1]
                break
        return betas, lams,

    def compute_group_norm(self, beta: torch.Tensor, group_size: List[int]):
        """compute the group norms of a beta"""
        group_norms = []
        for i in range(len(group_size)):
            start, end = self.find_group_index(group_size, i)
            group_norms.append(torch.norm(beta[start: end]))
        return group_norms

    def plot_path(self, result: dict, group_size: List[int] = None):
        """plot the solution path"""
        if group_size is None:
            group_size = self.group_size
        lams = list(result.keys())
        nz = compute_nonzeros(result[lams[-1]], group_size)[1]
        beta_norms = []
        for lam in lams:
            beta_norms.append(self.compute_group_norm(result[lam][1:], group_size))
        # lams = lams[1:]
        # beta_nonzero = []
        # for i in range(len(beta_norms)):
        #     beta_nonzero.append((np.array(beta_norms[i]) != 0).sum())
        # first_nonzero = beta_nonzero.index([i for i in beta_nonzero if i > 0][0])
        # if first_nonzero > 5:
        #     first_nonzero -= 5
        # else:
        #     first_nonzero = 0
        # lams = lams[first_nonzero:]
        # beta_norms = beta_norms[first_nonzero:]
        beta_norms = np.array(beta_norms)
        print(beta_norms.shape)
        plt.figure(figsize=(20, 5))
        plt.plot(lams, beta_norms[:, nz])
        plt.xlabel('Lambda')
        plt.ylabel('L2 norm of coefficients')
        plt.title('Group lasso path')
        plt.axis('tight')
        plt.legend(['beta' + str(i) for i in nz])
        plt.show()

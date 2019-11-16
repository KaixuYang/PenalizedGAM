import numpy as np
from patsy import dmatrix

class GAM:
    def __init__(self, learning_rate: float = 1, learning_rate_shrink: float = 0.8, tol: float = 1e-2,
                 data_class: str = 'regression', degree: int = 3, df: int = 5):
        self.learning_rate = learning_rate
        self.learning_rate_shrink = learning_rate_shrink
        self.tol = tol
        if data_class in ['regression', 'classification']:
            self.data_class = data_class
        else:
            raise ValueError("data_class must be regression or classification.")
        self.degree = degree
        self.df = df
        self.beta = None
        self.intercept = None
        self.beta_a = None
        self.intercept_a = None
        self.R = None  # right matrices of QR decomposition of basis matrix, used to orthogonalization.

    def basis_expansion_(self, x: np.array, df: int, degree: int):
        p = x.shape[1]
        if p > 1:
            for i in range(x.shape[1]):
                transformed_x = dmatrix(
                    "bs(x[:, i], df=df, degree=degree, include_intercept=False)",
                    {"train": x[:, i]}, return_type='matrix')
                try:
                    res = np.concatenate([res, np.array(transformed_x)[:, 1:]], axis=1)
                except:
                    res = np.array(transformed_x[:, 1:])
        else:
            res = dmatrix(
                "bs(x, df=df, degree=degree, include_intercept=intercept)", {"train": x}, return_type='matrix')
            res = np.array(res)[:, 1:]
        return res

    def compute_group_norm_(self, x: np.array, group_size: int):
        num_group = len(x) / group_size
        if num_group != int(num_group):
            raise ValueError("Cannot compute group norm, the length of x is not a multiple of group_size")
        num_group = int(num_group)
        group_norm = np.zeros([num_group, 1])
        for i in range(num_group):
            group_norm[i, 0] = np.linalg.norm(x[i*group_size: (i+1)*group_size])
        return group_norm

    def groupsum_(self, x: np.array, group_size: int):
        """
        computes the group sum of a matrix x (sum of l2-norm of each column)
        :param x: the parameter matrix
        :return: groupsum_, float
        """
        group_norm = self.compute_group_norm_(x, group_size)
        return np.sum(group_norm)

    def grouplassothres_(self, x: np.array, group_size: int, lam: float) -> np.array:
        """
        computes the group thresholding function
        :param x: the input parameter matrix
        :param lam: tuning parameter
        :return: group lasso penalized input
        """
        num_group = len(x) / group_size
        if num_group != int(num_group):
            raise ValueError("Cannot compute group norm, the length of x is not a multiple of group_size")
        num_group = int(num_group)
        for i in range(num_group):
            norm = np.linalg.norm(x[:, (i*group_size):(i+1)*group_size])
            if norm != 0:
                x[:, (i*group_size):(i+1)*group_size] = np.max(
                    [0, 1 - lam / norm]) * x[:, (i*group_size):(i+1)*group_size]
        return x

    def sigmoid_(self, x: np.array):
        return np.exp(x) / (1 + np.exp(x))

    def group_orthogonalize_(self, x: np.array, group_size: int):
        n, p = x.shape
        num_group = p / group_size
        if num_group != int(num_group):
            raise ValueError("Cannot compute group norm, the length of x is not a multiple of group_size")
        num_group = int(num_group)
        new_x = np.copy(x)
        r_matrices = []
        for i in range(num_group):
            q, r = np.linalg.qr(x[:, i*group_size:(i+1)*group_size])
            new_x[:, i*group_size:(i+1)*group_size] = q * np.sqrt(n)
            r_matrices.append(r / np.sqrt(n))
        self.R = r_matrices
        return new_x

    def group_orthogonalize_inverse_(self, x: np.array, group_size: int):
        num_group = len(self.R)
        if num_group != len(x) / group_size:
            raise ValueError('Group orthogonalization inverse failed, check input.')
        x_shape = x.shape
        x = x.reshape(-1)
        for i in range(num_group):
            x[i*group_size:(i+1)*group_size] = np.matmul(np.linalg.inv(self.R[i]), x[i*group_size:(i+1)*group_size])
        x = x.reshape(x_shape)
        return x

    def compute_gradient_reg_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float):
        n = x.shape[0]
        grad_intercept = np.sum(-2 * (y - np.matmul(x, beta) - intercept)) / n
        beta_norm = self.compute_group_norm_(beta, self.df)
        beta_norm = np.where(beta_norm > 0, beta_norm, 0.001)
        norm_derivative = (beta.reshape(-1, self.df) / beta_norm).reshape(-1, 1)
        grad_beta = -2 * np.matmul(x.T, y - np.matmul(x, beta) - intercept) / n + lam * norm_derivative
        return grad_intercept, grad_beta

    def compute_gradient_cla_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float):
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        grad_intercept = np.sum(-(y - self.sigmoid_(eta)) / n)
        beta_norm = self.compute_group_norm_(beta, self.df)
        beta_norm = np.where(beta_norm > 0, beta_norm, 0.001)
        norm_derivative = (beta.reshape(-1, self.df) / beta_norm).reshape(-1, 1)
        grad_beta = -np.matmul(x.T, y - self.sigmoid_(eta)) / n + lam * norm_derivative
        return grad_intercept, grad_beta

    def compute_loss_reg_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float):
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        return np.linalg.norm(y - eta) ** 2 / n + lam * self.groupsum_(beta, self.df)

    def compute_loss_cla_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float):
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        return -np.sum(y * eta - np.log(1 + np.exp(eta))) / n + lam * self.groupsum_(beta, self.df)

    def fit_(self, z: np.array, y: np.array, lam: float = 0, max_iters: int = 1000):
        z = self.group_orthogonalize_(z, self.df)
        beta = np.zeros([z.shape[1], 1]) + 0.1
        intercept = np.zeros([1, 1])
        print('Fitting starts, parameters initialized.')
        if self.data_class == 'regression':
            loss = self.compute_loss_reg_(z, y, beta, intercept, lam)
        else:
            loss = self.compute_loss_cla_(z, y, beta, intercept, lam)
        print('The initial loss is', loss)
        error = 10
        iters = 0
        while error > self.tol and iters <= max_iters:
            iters += 1
            learning_rate = self.learning_rate
            if self.data_class == 'regression':
                grad_intercept, grad_beta = self.compute_gradient_reg_(z, y, beta, intercept, lam)
            else:
                grad_intercept, grad_beta = self.compute_gradient_cla_(z, y, beta, intercept, lam)
            linesearch = 0
            while linesearch == 0:
                intercept_new = intercept - learning_rate * grad_intercept
                beta_new = beta - learning_rate * grad_beta
                if self.data_class == 'regression':
                    loss_new = self.compute_loss_reg_(z, y, beta_new, intercept_new, lam)
                else:
                    loss_new = self.compute_loss_cla_(z, y, beta_new, intercept_new, lam)
                if loss_new <= loss - learning_rate * 0.5 * (
                        np.linalg.norm(beta - beta_new) ** 2 + (intercept_new - intercept) ** 2):
                    linesearch = 1
                else:
                    learning_rate = learning_rate * self.learning_rate_shrink
            beta_new = self.grouplassothres_(beta_new, self.df, lam)
            if self.data_class == 'regression':
                loss_new = self.compute_loss_reg_(z, y, beta_new, intercept_new, lam)
            else:
                loss_new = self.compute_loss_cla_(z, y, beta_new, intercept_new, lam)
            if iters % 10 == 0:
                print('Iteration', iters, 'The loss is', loss_new)
            error = np.abs(loss - loss_new)
            beta = beta_new
            intercept = intercept_new
            loss = loss_new
        if iters < max_iters:
            print('Convergence checked with error', error, 'converged in', iters, 'steps.')
        else:
            print('Not converging, consider increasing the max_iters.')
        print('The loss is', loss)
        beta = self.group_orthogonalize_inverse_(beta, self.df)
        return intercept, beta

    def compute_weights_(self, beta: np.array, group_size: int):
        num_group = len(beta) // group_size
        weights = np.zeros([num_group, 1])
        for i in range(num_group):
            norm = np.linalg.norm(beta[i*group_size:(i+1)*group_size])
            if norm <= 1e-6:
                weights[i] = 1e6
            else:
                weights[i] = 1 / norm
        return weights

    def fit_grplasso(
            self, x: np.array, y: np.array, lam: float, max_iters: int = 1000, adaptivity: bool = False,
            lam_a: float = 0.1):
        z = self.basis_expansion_(x, self.df, self.degree)
        print('Starting fitting group lasso.')
        self.intercept, self.beta = self.fit_(z, y, lam, max_iters)
        print('Group lasso fit finished.')
        if adaptivity:
            print('Adaptivity is True, start fitting adaptive group lasso')
            weights = self.compute_weights_(self.beta, self.df)
            z_weighted = np.copy(z)
            for i in range(len(weights)):
                z_weighted[:, i*self.df:(i+1)*self.df] = z_weighted[:, i*self.df:(i+1)*self.df] / weights[i]
            self.intercept_a, self.beta_a = self.fit_(z_weighted, y, lam_a, max_iters)
            for i in range(len(weights)):
                self.beta_a[i*self.df:(i+1)*self.df] = self.beta_a[i*self.df:(i+1)*self.df] / weights[i]
            print('Adaptive group lasso fit finished.')

    def predict(self, x: np.array, coefs: str = 'gl'):
        if coefs not in ['gl', 'agl']:
            raise ValueError('Coefs is either gl for group lasso or agl for adaptive group lasso')
        if coefs == 'gl':
            if self.intercept is None or self.beta is None:
                raise ValueError('Models not trained, please run fit_grplasso first.')
            z = self.basis_expansion_(x, self.df, self.degree)
            y_pred = np.matmul(z, self.beta) + self.intercept
            if self.data_class == 'classification':
                y_pred = np.where(self.sigmoid_(y_pred) > 0.5, 1, 0)
        else:
            if self.intercept_a is None or self.beta_a is None:
                raise ValueError('Models not trained, please run fit_grplasso with adaptivity first.')
            z = self.basis_expansion_(x, self.df, self.degree)
            y_pred = np.matmul(z, self.beta_a) + self.intercept_a
            if self.data_class == 'classification':
                y_pred = np.where(self.sigmoid_(y_pred) > 0.5, 1, 0)
        return y_pred

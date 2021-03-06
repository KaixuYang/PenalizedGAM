import numpy as np
from patsy import dmatrix
from sklearn.model_selection import KFold

"""
This package has deprecated. Look at the new version GAM_torch.py
"""

class GAM:
    def __init__(self, learning_rate: float = 1, learning_rate_shrink: float = 0.8, tol: float = 1e-3,
                 data_class: str = 'regression', degree: int = 3, df: int = 5):
        """
        initialize the class
        :param learning_rate: step size of gradient descent
        :param learning_rate_shrink: shrink ratio of backtracking line search
        :param tol: tolerance
        :param data_class: 'regression' or 'classification'
        :param degree: degree of B-spline, default cubic B-spline
        :param df: df of knots, evenly distributed
        """
        self.learning_rate = learning_rate
        self.learning_rate_shrink = learning_rate_shrink
        self.tol = tol
        if data_class in ['regression', 'classification']:
            self.data_class = data_class
        else:
            raise ValueError("data_class must be regression or classification.")
        self.degree = degree
        self.df = df
        self.beta = None  # group lasso coefficients
        self.intercept = None  # group lasso intercept
        self.beta_a = None  # adaptive group lasso coefficients
        self.intercept_a = None  # adaptive group lasso intercept
        self.R = None  # right matrices of QR decomposition of basis matrix, used to orthogonalization.
        self.best_lam = None  # best lambda of group lasso
        self.best_lam_a = None  # best lambda of adaptive group lasso
        self.eta_without_g_cache = [-1, -1]  # cache for eta_without_g

    @staticmethod
    def basis_expansion_(x: np.array, df: int, degree: int) -> np.array:
        """
        perform a basis expansion of the design matrix, uses B-spline with evenly distributed knots.
        :param x: the design matrix
        :param df: df of B-spline, decides the number of knots with degree
        :param degree: degree of B-spline, 3 indicates cubic B-spline
        :return: the basis matrix
        """
        p = x.shape[1]
        if p > 1:
            res = None
            for i in range(x.shape[1]):
                transformed_x = dmatrix(
                    "bs(x[:, i], df=df, degree=degree, include_intercept=False)",
                    {"train": x[:, i]}, return_type='matrix')
                if res is None:
                    res = np.array(transformed_x[:, 1:])
                else:
                    res = np.concatenate([res, np.array(transformed_x)[:, 1:]], axis=1)
        else:
            res = dmatrix(
                "bs(x, df=df, degree=degree, include_intercept=intercept)", {"train": x}, return_type='matrix')
            res = np.array(res)[:, 1:]
        return res

    @staticmethod
    def compute_group_norm_(x: np.array, group_size: int) -> np.array:
        """
        computes the norm of each subgroup of a vector
        :param x: the input array
        :param group_size: the group size
        :return: an array with length of len(x) / group_size, where each entry is the norm of sub-vector
        """
        num_group = len(x) / group_size
        if num_group != int(num_group):
            raise ValueError("Cannot compute group norm, the length of x is not a multiple of group_size")
        num_group = int(num_group)
        group_norm = np.zeros([num_group, 1])
        for i in range(num_group):
            group_norm[i, 0] = np.linalg.norm(x[i * group_size: (i + 1) * group_size])
        return group_norm

    @staticmethod
    def groupsum_(self, x: np.array, group_size: int) -> np.array:
        """
        computes the group sum of a matrix x (sum of l2-norm of each column)
        :param self: class
        :param group_size: group size
        :param x: the parameter matrix
        :return: groupsum_, float
        """
        group_norm = self.compute_group_norm_(x, group_size)
        return np.sum(group_norm)

    def grouplassothres_(self, x: np.array, y: np.array, intercept: np.array, beta: np.array, g: int, group_size: int,
                         lam: float) -> bool:
        """
        computes the group thresholding function
        :param group_size: number of variables in each group
        :param g: the group being applied soft thresholding
        :param beta: the beta vector
        :param intercept: the intercept
        :param y: the response
        :param x: the input parameter matrix
        :param lam: tuning parameter
        :return: Zero (True) or not zero (False)
        """
        if self.eta_without_g_cache[0] != g:
            self.eta_without_g_cache[0] = g
            x_without_g = np.delete(x, range(g * group_size, (g + 1) * group_size), 1)
            beta_without_g = np.delete(beta, range(g * group_size, (g + 1) * group_size), 0)
            eta_without_g = np.matmul(x_without_g, beta_without_g) + intercept
            self.eta_without_g_cache[1] = eta_without_g
        else:
            eta_without_g = self.eta_without_g_cache[1]
        if self.data_class == 'classification':
            left = np.linalg.norm(np.matmul(
                x[:, g * group_size:(g + 1) * group_size].T, y - self.sigmoid_(eta_without_g))) / x.shape[0]
        else:
            left = np.linalg.norm(np.matmul(
                x[:, g * group_size:(g + 1) * group_size].T, y - eta_without_g)) / x.shape[0]
        if left <= lam:
            return True
        else:
            return False

    @staticmethod
    def sigmoid_(x: np.array) -> np.array:
        """
        computes the sigmoid function
        :param x: input array
        :return: sigmoid of the input
        """
        return np.exp(x) / (1 + np.exp(x))

    def group_orthogonalize_(self, x: np.array, group_size: int) -> np.array:
        """
        implements group-wise orthogonalization of the basis matrix
        :param x: the basis matrix
        :param group_size: the group size
        :return: the group-wisely orthogonalized basis matrix
        """
        n, p = x.shape
        num_group = p / group_size
        if num_group != int(num_group):
            raise ValueError("Cannot compute group norm, the length of x is not a multiple of group_size")
        num_group = int(num_group)
        new_x = np.copy(x)
        r_matrices = []
        for i in range(num_group):
            q, r = np.linalg.qr(x[:, i * group_size:(i + 1) * group_size])
            new_x[:, i * group_size:(i + 1) * group_size] = q * np.sqrt(n)
            r_matrices.append(r / np.sqrt(n))
        self.R = r_matrices
        return new_x

    def group_orthogonalize_inverse_(self, x: np.array, group_size: int) -> np.array:
        """
        inverse transformation of the group-wise orthogonalization on the coefficients
        :param x: the coefficients
        :param group_size: the group size
        :return: the inverse transformed coefficients
        """
        num_group = len(self.R)
        if num_group != len(x) / group_size:
            raise ValueError('Group orthogonalization inverse failed, check input.')
        x_shape = x.shape
        x = x.reshape(-1)
        for i in range(num_group):
            x[i * group_size:(i + 1) * group_size] = np.matmul(np.linalg.inv(self.R[i]),
                                                               x[i * group_size:(i + 1) * group_size])
        x = x.reshape(x_shape)
        return x

    @staticmethod
    def compute_grad_intercept_reg_(x: np.array, y: np.array, beta: np.array, intercept: np.array) -> np.array:
        """
        computes the gradient of the intercept in the regression mode
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficents
        :param intercept: the intercept
        :return: the gradient of the intercept
        """
        n = x.shape[0]
        grad_intercept = np.sum(-2 * (y - np.matmul(x, beta) - intercept)) / n
        return grad_intercept

    @staticmethod
    def compute_grad_intercept_cla_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array) -> np.array:
        """
        computes the gradient of the intercept in the classification mode
        :param self: the class
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficients
        :param intercept: the intercept
        :return: the gradient of the intercept
        """
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        grad_intercept = np.sum(-(y - self.sigmoid_(eta)) / n)
        return grad_intercept

    def compute_grad_beta_reg_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float,
                               g: int) -> np.array:
        """
        computes the gradient of the gth group coefficients in the regression mode
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficients
        :param intercept: the intercept
        :param lam: the lambda
        :param g: the group index
        :return: the gradient of the gth group of coefficients
        """
        n = x.shape[0]
        beta_norm = np.linalg.norm(beta[g * self.df:(g + 1) * self.df])
        beta_norm = np.where(beta_norm > 0, beta_norm, 0.001)
        norm_derivative = (beta[g * self.df:(g + 1) * self.df].reshape(-1) / beta_norm).reshape(-1, 1)
        grad_beta = -2 * np.matmul(x[:, g * self.df:(g + 1) * self.df].T, y - np.matmul(x, beta) - intercept) / n \
            + lam * norm_derivative
        return grad_beta

    def compute_grad_beta_cla_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float,
                               g: int) -> np.array:
        """
        computes the gradient of the gth group coefficients in the classification mode
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficients
        :param intercept: the intercept
        :param lam: the lambda
        :param g: the group index
        :return: the gradient of the gth group of coefficients
        """
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        beta_norm = np.linalg.norm(beta[g * self.df:(g + 1) * self.df])
        beta_norm = np.where(beta_norm > 0, beta_norm, 0.001)
        norm_derivative = (beta[g * self.df:(g + 1) * self.df].reshape(-1) / beta_norm).reshape(-1, 1)
        grad_beta = -np.matmul(x[:, g * self.df:(g + 1) * self.df].T,
                               y - self.sigmoid_(eta)) / n + lam * norm_derivative
        return grad_beta

    def compute_loss_reg_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float) -> float:
        """
        computes the loss function in the regression mode
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficients
        :param intercept: the intercept
        :param lam: the lambda
        :return: the loss function value
        """
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        return np.linalg.norm(y - eta) ** 2 / n + lam * self.groupsum_(self, beta, self.df)

    def compute_loss_cla_(self, x: np.array, y: np.array, beta: np.array, intercept: np.array, lam: float) -> float:
        """
        computes the loss function in the classification mode
        :param x: the basis matrix
        :param y: the response
        :param beta: the coefficients
        :param intercept: the intercept
        :param lam: the lambda
        :return: the loss function value
        """
        n = x.shape[0]
        eta = np.matmul(x, beta) + intercept
        return -np.sum(y * eta - np.log(1 + np.exp(eta))) / n + lam * self.groupsum_(self, beta, self.df)

    def fit_(self, z: np.array, y: np.array, lam: float = 0, max_iters: int = 1000) -> [np.array, np.array]:
        """
        fits the group lasso GAM with block-coordinate descent algorithm, internal function
        :param z: the basis matrix
        :param y: the response
        :param lam: the tuning parameter
        :param max_iters: maximum number of iterations
        :return: the fitted intercept and coefficients
        """
        z = self.group_orthogonalize_(z, self.df)
        beta = np.zeros([z.shape[1], 1])
        intercept = np.zeros([1, 1])
        beta_new = np.copy(beta)
        intercept_new = np.copy(intercept)
        print('Fitting starts, parameters initialized.')
        if self.data_class == 'regression':
            loss = self.compute_loss_reg_(z, y, beta, intercept, lam)
        else:
            loss = self.compute_loss_cla_(z, y, beta, intercept, lam)
        print('The initial loss is', loss)
        loss_new = loss
        error = 10
        iters = 0
        while error > self.tol and iters <= max_iters:
            iters += 1
            """start minimizing the loss function with respect to the intercept"""
            learning_rate = self.learning_rate
            if self.data_class == 'regression':
                grad_intercept = self.compute_grad_intercept_reg_(z, y, beta, intercept)
            else:
                grad_intercept = self.compute_grad_intercept_cla_(self, z, y, beta, intercept)
            ls_intercept = 0  # Line search indicator for intercept
            while ls_intercept == 0:
                intercept_new = intercept - learning_rate * grad_intercept
                if self.data_class == 'regression':
                    loss_new = self.compute_loss_reg_(z, y, beta_new, intercept_new, lam)
                else:
                    loss_new = self.compute_loss_cla_(z, y, beta_new, intercept_new, lam)
                if loss_new <= loss - learning_rate * 0.5 * (intercept_new - intercept) ** 2 or learning_rate < 0.001:
                    ls_intercept = 1
                else:
                    learning_rate = learning_rate * self.learning_rate_shrink
            intercept = intercept_new  # intercept updated
            loss_intercept = loss_new
            """Start iterating among the groups"""
            num_group = len(beta) // self.df
            for g in range(num_group):
                """shrunk beta g to zero if criterion satisfied"""
                if self.grouplassothres_(z, y, intercept, beta, g, self.df, lam):
                    beta[g * self.df:(g + 1) * self.df] = np.zeros([self.df, 1])
                else:
                    """If criterion not satisfied, minimizing loss function with respect to beta g"""
                    error_beta = 10
                    while error_beta > self.tol:
                        ls_beta = 0
                        learning_rate = self.learning_rate
                        if self.data_class == 'regression':
                            grad_beta_g = self.compute_grad_beta_reg_(z, y, beta, intercept, lam, g)
                        else:
                            grad_beta_g = self.compute_grad_beta_cla_(z, y, beta, intercept, lam, g)
                        while ls_beta == 0:
                            beta_new = np.copy(beta)
                            beta_new[g * self.df:(g + 1) * self.df] = beta_new[g * self.df:(g + 1) * self.df] - \
                                learning_rate * grad_beta_g
                            if self.data_class == 'regression':
                                loss_new = self.compute_loss_reg_(z, y, beta_new, intercept, lam)
                            else:
                                loss_new = self.compute_loss_cla_(z, y, beta_new, intercept, lam)
                            if loss_new <= loss_intercept - learning_rate * 0.5 * np.linalg.norm(beta_new - beta) ** 2 \
                                    or learning_rate < 0.001:
                                ls_beta = 1
                            else:
                                learning_rate = learning_rate * self.learning_rate_shrink
                        beta = beta_new
                        error_beta = np.abs(loss_intercept - loss_new)
                        loss_intercept = loss_new
            error = np.abs(loss_intercept - loss)
            loss = loss_intercept
            if iters % 10 == 0:
                print('Iteration', iters, 'The loss is', loss_new)
        if iters < max_iters:
            print('Convergence checked with error', error, 'converged in', iters, 'steps.')
        else:
            print('Not converging, consider increasing the max_iters.')
        print('The loss is', loss_new)
        beta = self.group_orthogonalize_inverse_(beta, self.df)
        return intercept, beta

    @staticmethod
    def compute_weights_(beta: np.array, group_size: int) -> np.array:
        """
        computes the adaptive group lasso weights from the fitted coefficients
        :param beta: the fitted coefficients
        :param group_size: the group size
        :return: the weights
        """
        num_group = len(beta) // group_size
        weights = np.zeros([num_group, 1])
        for i in range(num_group):
            norm = np.linalg.norm(beta[i * group_size:(i + 1) * group_size])
            if norm <= 1e-6:
                weights[i] = 1e6
            else:
                weights[i] = 1 / norm
        return weights

    @staticmethod
    def agl_submatrix(
            x: np.array, beta: np.array, weights: np.array, group_size: int) -> (list, np.array, np.array):
        """
        extracts the nonzero submatrix for adaptive group lasso
        :param x: the basis matrix
        :param beta: the group lasso estimated coefficients
        :param weights: the weights
        :param group_size: the group size
        :return: the nonzero index, nonzero weights and nonzero submatrix x
        """
        nonzero = []
        weights_nonzero = []
        x_nonzero = None
        for i in range(len(weights)):
            if np.linalg.norm(beta[i*group_size:(i+1)*group_size]) > 0:
                nonzero.append(i)
                weights_nonzero.append(weights[i])
                if x_nonzero is None:
                    x_nonzero = x[:, i*group_size:(i+1)*group_size]
                else:
                    x_nonzero = np.concatenate([x_nonzero, x[:, i*group_size:(i+1)*group_size]], axis=1)
        return nonzero, np.array(weights_nonzero), x_nonzero

    def fit(self, x: np.array, y: np.array, lam: float, max_iters: int = 1000, adaptivity: bool = False,
            lam_a: float = 0.1) -> None:
        """
        fits the model with a use specified lambda
        :param x: the design matrix
        :param y: the response
        :param lam: the lambda for group lasso
        :param max_iters: the maximum number of iterations
        :param adaptivity: True for fitting adaptive group lasso GAM after fitting group lasso GAM
        :param lam_a: the lambda for adaptive group lasso
        :return: None
        """
        z = self.basis_expansion_(x, self.df, self.degree)
        print('Starting fitting group lasso.')
        self.intercept, self.beta = self.fit_(z, y, lam, max_iters)
        print('Group lasso fit finished.')
        if adaptivity:
            print('Adaptivity is True, start fitting adaptive group lasso')
            weights = self.compute_weights_(self.beta, self.df)
            nonzero, weights, z = self.agl_submatrix(z, self.beta, weights, self.df)
            z_weighted = np.copy(z)
            for i in range(len(weights)):
                z_weighted[:, i * self.df:(i + 1) * self.df] = z_weighted[:, i * self.df:(i + 1) * self.df] / weights[i]
            self.intercept_a, beta_a = self.fit_(z_weighted, y, lam_a, max_iters)
            for i in range(len(weights)):
                beta_a[i * self.df:(i + 1) * self.df] = beta_a[i * self.df:(i + 1) * self.df] / weights[i]
            self.beta_a = np.zeros(self.beta.shape)
            for i in range(len(self.beta) // self.df):
                if i in nonzero:
                    i_index = nonzero.index(i)
                    self.beta_a[i*self.df:(i+1)*self.df] = beta_a[i_index*self.df:(i_index+1)*self.df]
            print('Adaptive group lasso fit finished.')

    def fit_cv(self, x: np.array, y: np.array, max_iters: int = 1000, lams: list = None, adaptivity: bool = False,
               lam_as: list = None, cvfold: int = 5) -> None:
        """
        fits the GAM with cross validation
        :param x: the design matrix
        :param y: the response
        :param max_iters: the maximum number of iterations
        :param lams: lambda candidates for group lasso
        :param adaptivity: True for fitting adaptive group lasso
        :param lam_as: lambda candidates for adaptive group lasso
        :param cvfold: number of cross validation folds
        :return: None
        """
        if lams is None:
            lams = [1e-3, 1e-2, 2e-2, 3e-2, 5e-2, 8e-2, 0.1, 0.15, 0.2]
        z = self.basis_expansion_(x, self.df, self.degree)
        print('Starting fitting group lasso cv, parameters initialized.')
        cv_scores = []
        for lam in lams:
            cv_score = []
            cv = KFold(n_splits=cvfold)
            for train, test in cv.split(z):
                z_train = z[train, :]
                z_test = z[test, :]
                y_train = y[train]
                y_test = y[test]
                intercept, beta = self.fit_(z_train, y_train, lam, max_iters)
                y_pred = np.matmul(z_test, beta) + intercept
                if self.data_class == 'classification':
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                cv_score.append(np.mean((y_pred - y_test) ** 2))
            cv_scores.append(np.mean(cv_score))
            print('CV with lambda', lam, 'finished, average score', cv_scores[-1])
        lam = lams[cv_scores.index(min(cv_scores))]
        self.best_lam = lam
        self.intercept, self.beta = self.fit_(z, y, lam, max_iters)
        print('Group lasso cv finished.')
        if adaptivity:
            if lam_as is None:
                lam_as = [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2]
            weights = self.compute_weights_(self.beta, self.df)
            nonzero, weights, z = self.agl_submatrix(z, self.beta, weights, self.df)
            print('Start adaptive group lasso CV, parameters initialized')
            z_weighted = np.copy(z)
            for i in range(len(weights)):
                z_weighted[:, i * self.df:(i + 1) * self.df] = z_weighted[:, i * self.df:(i + 1) * self.df] / weights[i]
            cv_scores = []
            for lam in lam_as:
                cv_score = []
                cv = KFold(n_splits=cvfold)
                for train, test in cv.split(z_weighted):
                    z_train = z_weighted[train, :]
                    z_test = z_weighted[test, :]
                    y_train = y[train]
                    y_test = y[test]
                    intercept_a, beta_a = self.fit_(z_train, y_train, lam, max_iters)
                    for i in range(len(weights)):
                        beta_a[i * self.df:(i + 1) * self.df] = beta_a[i * self.df:(i + 1) * self.df] / weights[i]
                    y_pred = np.matmul(z_test, beta_a) + intercept_a
                    if self.data_class == 'classification':
                        y_pred = np.where(y_pred > 0, 1, 0)
                    cv_score.append(np.mean((y_pred - y_test) ** 2))
                cv_scores.append(np.mean(cv_score))
                print('CV with lambda', lam, 'finished, average score is', cv_scores[-1])
            lam_a = lams[cv_scores.index(min(cv_scores))]
            self.best_lam_a = lam_a
            self.intercept_a, beta_a = self.fit_(z_weighted, y, lam_a, max_iters)
            for i in range(len(weights)):
                beta_a[i * self.df:(i + 1) * self.df] = beta_a[i * self.df:(i + 1) * self.df] / weights[i]
            self.beta_a = np.zeros(self.beta.shape)
            for i in range(len(self.beta) // self.df):
                if i in nonzero:
                    i_index = nonzero.index(i)
                    self.beta_a[i*self.df:(i+1)*self.df] = beta_a[i_index*self.df:(i_index+1)*self.df]
            print('Adaptive group lasso cv finished.')

    def predict(self, x: np.array, coefs: str = 'gl') -> np.array:
        """
        makes prediction for a design matrix x
        :param x: the design matrix
        :param coefs: 'gl' for using group lasso or 'agl' for using adaptive group lasso
        :return: the prediction
        """
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

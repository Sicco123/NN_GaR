import numpy as np
from scipy.optimize import linprog


class quantile_regression():
    def __init__(self, quantile_levels, solver, solver_options = None):

        self.quantile_levels = quantile_levels
        self.solver = solver
        self.solver_options = solver_options

    def fit(self, X, y, quantile_level):
        # https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem
        n_params = X.shape[1]
        n_indices = X.shape[0]
        ones = np.ones( n_indices)

        c = np.concatenate(
            [
                np.full(2 * n_params, 0),
                ones.T * quantile_level,
                ones.T * (1 - quantile_level),
            ]
        )

        eye = np.eye(n_indices)
        A_eq = np.concatenate([X, -X, eye, -eye], axis=1)
        b_eq = y


        # make default solver more stable
        if self.solver_options is None and self.solver == "interior-point":
            solver_options = {"lstsq": True}
        else:
            solver_options = self.solver_options

        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            method=self.solver,
            options=solver_options,
        )
        solution = result.x

        params = solution[:n_params] - solution[n_params: 2 * n_params]
        return params

    def fit_multiple(self, X, y):
        vfit = np.vectorize(self.fit, otypes=[list], excluded=['X', 'y'])
        out = vfit(X = X, y = y, quantile_level = self.quantile_levels)

        self.vparams = np.row_stack(out)
        return self

    def predict(self, X, params):
        reg_quantiles = params @ X.T
        return reg_quantiles

    def predict_multiple(self, X):
        reg_quantiles =  X @ self.vparams.T
        return reg_quantiles






#
# n_params = X.shape[1]
#         n_indices = X.shape[0]
#         identity = np.ones( n_indices)
#
#         c = np.concatenate(
#             [
#                 np.full(2 * n_params, fill_value=alpha),
#                 identity.T * self.quantile_level,
#                 identity.T * (1 - self.quantile_level),
#             ]
#         )
#
#         print(c.shape)
#         eye = np.eye(n_indices)
#         print(X.shape)
#         print(eye.shape)
#         A_eq = np.concatenate([X, -X, eye, -eye], axis=1)
#         print(A_eq.shape)
#         b_eq = y
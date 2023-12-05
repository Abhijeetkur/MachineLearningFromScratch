import numpy as np

class LinearRegression1:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
            """
            Here's the basic syntax of numpy.linalg.lstsq():
                    numpy.linalg.lstsq(a, b, rcond='warn')
                    Parameters:
                    a: Coefficient matrix. It represents the system of linear equations.
                    b: Ordinate values. It represents the observed values.
                    rcond: Relative condition number. It is used to determine the effective rank of a. If rcond is set to None,             the default value, it is automatically determined based on machine precision.

                    The function returns a tuple (x, residuals, rank, s), where:

                    x: The solution (the coefficients) (coefficients will contain the solution to the linear least squares problem, which corresponds to the intercept and slope in the context of linear regression).
                    residuals: The sum of squared residuals.
                    rank: The effective rank of a.
                    s: Singular values of a.
"""
        self.coef = np.linalg.lstsq(X, y, rcond=-1)[0]

    def predict(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return np.dot(X, self.coef)

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        self._new_X = np.concatenate((intercept, X), axis=1)
        return self._new_X
import numpy as np
from numpy import cos
from numpy import sqrt
from numpy import square


class CostFunctions(object):
    """This class contains all the cost functions that would be used
    to test the optimizers"""
    def griewank(self, params):
        ind_lst = np.arange(params.shape[0]) + 1.
        first_term = 1 + np.sum(square(params)) * (1. / 4000)
        second_term = np.product(cos(params / sqrt(ind_lst)))
        return first_term - second_term

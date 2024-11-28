"""

This module provides a flexible framework for defining and applying constraints to neural networks.
It includes support for custom constraint definitions, enabling constraints on linear combinations
of arbitrary partial derivatives. These constraints can be applied to both discrete and continuous
domains and support equality and inequality formulations.

Custom constraints can be defined by subclassing the provided base classes. For example,
see the implementations of CIEQC, CEQC, DIEQC, and DEQC.

Version: 3.1
Author: Jannick Stranghöner
"""

__version__ = '3.1'
__author__ = 'Jannick Stranghöner'

from abc import ABC
from typing import List, Callable, Optional

import numpy as np

from regions import SamplingRegion


class BaseConstraint(ABC):
    """
    Base class for all constraints. Handles the initialization of multiple parent classes
    through a unified constructor, which invokes all other parent class constructors with the
    appropriate named arguments, so you do not have to repeat all the keyword arguments from
    the, possibly many, parent classes. Sadly there is no way of signaling this behavior to the
     static analysis, so you lose constructor auto completion (at least I have not found a way
     to do this in PyCharm / VSCode).

    Args:
        weight (float): The weight of the constraint in the loss function. Default is 1.0.
        label (str): A label for identifying the constraint. Default is an empty string.
        lagrange_alpha (float): Learning rate for Lagrange multiplier updates. Default is 1e-3.
        **kwargs: Additional arguments passed to parent classes.
    """
    def __init__(self, weight: float = 1.0, label: str = '', lagrange_alpha: float = 1e-3, **kwargs):
        self.inp_dim = None
        self.weight = weight
        self.label = label
        self.lagrange_alpha = lagrange_alpha

        # a constraint has many super constructors by design, and each gets called
        # with respective named arguments by this method
        self.initialize_child_class_from_kwargs(**kwargs)

    def initialize_child_class_from_kwargs(self, **kwargs):
        """
        Calls the constructors of parent classes with their respective arguments.

        Args:
            **kwargs: Keyword arguments for the parent class constructors.
        """
        for cls in self.__class__.__bases__:
            if cls is BaseConstraint:
                continue

            # call each super constructor with respective arguments
            kwargs_sub = {}
            for var_name in cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]:
                if var_name in kwargs:
                    kwargs_sub[var_name] = kwargs[var_name]
            cls.__init__(self, **kwargs_sub)


class DiscreteConstraint(ABC):
    """
    Represents constraints defined over discrete variables.

    Args:
        u (np.ndarray): A discrete set of variable values.
    """
    def __init__(self, u):
        self.u = u


class ContinuousConstraint(ABC):
    """
    Represents constraints defined over continuous regions.

    Args:
        region (SamplingRegion): The sampling region for continuous variables.
        samples_per_iteration (int): Number of samples added per optimization iteration. Default is 1.
        test_samples_per_iteration (int): Number of test samples per evaluation. Default is 1000.
        satisfaction_threshold (float): Threshold for considering a constraint satisfied. Default is 0.998.
        max_pool_size (int): Maximum size of the sample pool. Default is 25.
        max_test_value (float): Optional upper bound for the test value, which is used for constraint evaluation. If it is none, an heuristic is used.
        min_test_value (float): Optional lower bound for the test value, which is used for constraint evaluation. If it is none, an heuristic is used.
    """
    def __init__(self, region: SamplingRegion,
                 samples_per_iteration: int = 1,
                 test_samples_per_iteration: int = 1000,
                 satisfaction_threshold: float = 0.998,
                 max_pool_size: int = 25,
                 max_test_value: Optional[float] = None,
                 min_test_value: Optional[float] = None):
        self.region = region
        self.samples_per_iteration = samples_per_iteration
        self.test_samples_per_iteration = test_samples_per_iteration
        self.satisfaction_threshold = satisfaction_threshold
        self.max_pool_size = max_pool_size
        self.max_test_value = max_test_value
        self.min_test_value = min_test_value
        self.u = np.array([[]])

    def draw_test_samples(self, sample_size: Optional[int] = None):
        """
        Draws test samples from the region.

        Args:
            sample_size (int, optional): Number of test samples to draw. Defaults to `test_samples_per_iteration`.
        Returns:
            np.ndarray: Array of test samples.
        """
        if sample_size is None:
            sample_size = self.test_samples_per_iteration
        return self.region.sample(sample_size)


class InequalityConstraint(ABC):
    """
    Represents inequality constraints.

    Args:
        max_value (float): Upper bound for the constraint. Default is np.inf.
        min_value (float): Lower bound for the constraint. Default is -np.inf.
    """
    def __init__(self, max_value: float = np.inf, min_value: Optional[float] = -np.inf):
        self.min_value = min_value
        self.max_value = max_value


class EqualityConstraint(ABC):
    """
    Represents equality constraints.

    Args:
        value (float): Target value for the constraint.
        eps (float): Tolerance for equality. Default is 1e-3.
    """
    def __init__(self, value: float, eps: Optional[float] = 1e-3):
        self.value = value
        self.eps = eps


class LinearConstraint(ABC):
    """
    Represents linear constraints based on partial derivatives.

    While this class allows you to to define any linear combination of any partial derivative of a multivariate function, its syntax is by
    no means straightforward and best explained using examples:

    ================================================================================================================

    given a CELM f that realizes a R3 -> R2 function (f_(n) corresponds f_(n)(x) for readability)

    1.)
    partials=[[[], []]], factors=[[1, 0]] | corresponds to C = 1 * f_0 + 0 * f_1

    Explanation: The innermost brackets of 'partials' correspond to the partial derivatives of a particular output neuron.
    They are empty, which corresponds to the zeroth derivative - the function itself. The second innermost brackets of 'partials' and the
    innermost brackets of 'factors' correspond to a weighted sum over the output neurons. Therefore, there are exactly two empty brackets
    in the second innermost brackets of partials and exactly two factors in the innermost bracktes of 'factors'.
    The outermost brackets correspond to a "sum of weighted sums", which is only rarely necessary. It enables constraints over
    scalar products in vector fields, for example.

    2.) partials=[[[0, 1], []]], factors=[[1, 0]] | corresponds to C = 1 * f_0 / (dx_0 * d_x1) + 0 * f_1

    Explanation: Here, the innermost brackets are not empty, and correspond to two partial derivative of the first output neurons w.r.t.
    to the first and the second input neuron.

    3.)
    partials=[[[1], [2]]], factors=[[0.5, 0.5]] | corresponds to C = 0.5 * df_0 / dx_1 + 0.5 * df_1 / dx_2

    Explanation: This one should be straightforward, each output neurons has its own partial derivative and its own factor.

    4.)
    partials=[[[], []], factors=[g] | corresponds to C = g(x)_0 * f_0 + g(x)_1 * f_1, with g being another R3 -> R2 function
    Explanation: The CELM formulation enables factors to be arbitrary (potentially non-linear) functions of the input - the constraint as
    a whole remains linear in terms of the function output. g can even be another CELM!

    5.)
    partials=[[[], []], [[0,1], []]], factors=[g, [2, 0]] |
    corresponds to C = g(x)_0 * f_0 + g(x)_1 * f_1 + 2 * df_0^2 / (dx_0 * d_x1)

    Explanation: Here, everything comes together.

    Args:
        partials (List): List of partial derivatives for the constraint.
        factors (List): List of factors corresponding to the partial derivatives.
    """
    def __init__(self, partials: List, factors: List):
        self.partials = partials
        self.factors = factors


class NonlinearConstraint(ABC):
    """
    Represents nonlinear constraints.

    Args:
        func (Callable): A callable function defining the constraint.
    """
    def __init__(self, func: Callable):
        self.func = func


class CIEQC(BaseConstraint, LinearConstraint, ContinuousConstraint, InequalityConstraint):
    "A linear continuous inequality constraint."
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class CEQC(BaseConstraint, LinearConstraint, ContinuousConstraint, EqualityConstraint):
    "A linear continuous equality constraint."
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class DIEQC(BaseConstraint, LinearConstraint, DiscreteConstraint, InequalityConstraint):
    "A linear discrete inequality constraint."
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)


class DEQC(BaseConstraint, LinearConstraint, DiscreteConstraint, EqualityConstraint):
    "A linear discrete equality constraint."
    def __init__(self, *args, **kwargs):
        BaseConstraint.__init__(self, *args, **kwargs)

class ObjFct:
    """
    Represents an objective function defined by partial derivatives and factors.

    Args:
        partials (List): List of partial derivatives.
        factors (List): List of factors corresponding to the partial derivatives.
        inp_dim (int): Input dimension of the function. Default is 1.
    """
    def __init__(self, partials, factors, inp_dim=1):
        self.partials = partials
        self.factors = factors
        self.inp_dim = inp_dim


if __name__ == "__main__":
    con = DIEQC(2, partials=[[[0]]], factors=[[1]], u=[[0]])

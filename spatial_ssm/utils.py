"""Utility functions for spatial State Space Models."""

from abc import ABC, abstractmethod

import numpy as np

from scipy.spatial import distance_matrix


class Kernel(ABC):
    """A simple kernel base class for spatial covariance functions."""

    @abstractmethod
    def __call__(self, x, y):
        """Calculate the kernel matrix between two sets of points.

        Args:
            x: The first set of points.
            y: The second set of points.

        Returns:
            The kernel matrix between the two sets of points.
        """


class Exponential(Kernel):
    """The exponential kernel.

    Attributes:
        l (float): The length scale of the kernel.
        sigma (float): The variance of the kernel.
    """

    def __init__(self, l: float = 1.0, sigma: float = 1.0):
        """Initialize the exponential kernel.

        Args:
            l: The length scale of the kernel. Defaults to 1.0.
            sigma: The variance of the kernel. Defaults to 1.0.

        Raises:
            ValueError: If the length scale or variance are not positive.
        """

        if l <= 0:
            raise ValueError("Length scale must be positive.")
        if sigma <= 0:
            raise ValueError("Variance must be positive.")

        self.l = l
        self.sigma = sigma

    def __call__(self, x, y):
        return self.sigma**2 * np.exp(-distance_matrix(x, y) / self.l)


class RBF(Kernel):
    """The radial basis function kernel.

    Attributes:
        l: The length scale of the kernel.
        sigma: The variance of the kernel
    """

    def __init__(self, l: float = 1.0, sigma: float = 1.0):
        """Initialize the RBF kernel.

        Args:
            l: The length scale of the kernel. Defaults to 1.0.
            sigma: The variance of the kernel. Defaults to 1.0.

        Raises:

            ValueError: If the length scale or variance are not positive.
        """

        if l <= 0:
            raise ValueError("Length scale must be positive.")
        if sigma <= 0:
            raise ValueError("Variance must be positive.")

        self.l = l
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.sigma**2 * np.exp(-distance_matrix(x, y) ** 2 / (2 * self.l**2))

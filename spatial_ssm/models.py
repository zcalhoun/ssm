"""Spatial state space models

This module contains implementations of spatial state space models.

A list of models:
    - Spatial Random Walk

"""

import numpy as np
from scipy.linalg import block_diag

from sklearn.gaussian_process import kernels

from .kalman import LinearGaussianSSM
from .base import SpatialState


class SpatialRandomWalk(LinearGaussianSSM):
    """The Spatial Random Walk model.

    This model implements the state space model described by

        y_t = \mu_t + U_t + \epsilon_t
            \epsilon_t \sim N(0, \Sigma_{\epsilon})
        \mu_{t} = \mu_{t-1} + \eta_t
            \eta_t \sim N(0, \Sigma_{\eta})
        U_t = \alpha U_{t-1} + \omega_t
            \omega_t \sim N(0, \Sigma_{\omega})

    """

    def __init__(
        self,
        mean_noise: float,
        spatial_kernel: kernels.Kernel,
        spatiotemporal_corr: float,
        observation_noise: float,
        obs_coords: int,
        initial_state: SpatialState,
    ):
        """Initialize the Spatial Random Walk model.

        This init function just takes in the special parameters of the spatial
        random walk model, and converts them into a LinearGaussianSSM.

        Args:
            mean_noise: The mean noise of the model.
            spatial_kernel: The spatial kernel.
            spatiotemporal_corr: the spatiotemporal correlation parameter.
            observation_noise: The observation noise.
            obs_coords: The observation coordinates.
            initial_state: The initial state of the model.

        """
        self.mean_noise = mean_noise
        self.kernel = spatial_kernel
        self.alpha = spatiotemporal_corr
        self.num_obs = len(obs_coords)
        self.obs_noise = observation_noise

        # Create the transition matrix
        observation_noise = np.eye(self.num_obs) * observation_noise
        transition_noise = block_diag(
            mean_noise,
            self.kernel(initial_state.coords, initial_state.coords)
            @ (np.eye(len(initial_state.coords)) - spatiotemporal_corr**2),
        )

        transition_matrix = block_diag(
            1,
            self.alpha * np.eye(len(initial_state.coords)),
        )

        k_yx = self.kernel(obs_coords, initial_state.coords)
        k_xx = self.kernel(initial_state.coords, initial_state.coords)
        k_xx_inv = np.linalg.inv(k_xx)

        observation_matrix = np.hstack(
            (
                np.ones((self.num_obs, 1)),
                k_yx @ k_xx_inv,
            )
        )

        super().__init__(
            transition_matrix,
            observation_matrix,
            transition_noise,
            observation_noise,
            initial_state,
        )

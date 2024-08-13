"""Spatial state space models

This module contains implementations of spatial state space models.

A list of models:
    - Spatial Random Walk

"""

import numpy as np
from scipy.linalg import block_diag

from sklearn.gaussian_process import kernels

from .kalman import LinearGaussianSSM
from .base import SpatialState, State, DiffuseState


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


class LocalLinearTrend(LinearGaussianSSM):
    """Local Linear Trend model

    Contains the code to initialize a local linear trend model.

    An optional period parameter can be passed into the model. If none is passed,
    then there is assumed to be no periodicity.

    Attributes:
        mean_noise: The mean noise of the model.
        trend_noise: The trend noise of the model.

    """

    def __init__(
        self,
        mean_noise: float,
        trend_noise: float = 1.0,
        observation_noise: float = 1.0,
        initial_state: State | None = None,
        period: int | None = None,
        period_noise: float = 1.0,
    ):
        """Initialize the Local Linear Trend model.

        Args:
            mean_noise: The observation noise.
            trend_noise: The transition noise.
            observation_noise: The initial state of the model.
            initial_state: The initial state of the model. Defaults to None.
            period: The periodicity of the model. Defaults to None.
            period_noise: The noise of the period. Defaults to 1.0.
        """
        self.period = period

        transition_noise = self._init_transition_error(
            mean_noise, trend_noise, period_noise
        )

        observation_noise = np.array([[observation_noise]])

        transition_matrix = self._init_transition_matrix()
        observation_matrix = self._init_observation_matrix()

        if initial_state is None:
            initial_state = self._init_diffuse_state()
        super().__init__(
            transition_matrix,
            observation_matrix,
            transition_noise,
            observation_noise,
            initial_state,
        )

    def _init_diffuse_state(self) -> DiffuseState:
        """Initialize the diffuse state so the model can learn"""

        if self.period is None:
            return DiffuseState(np.zeros((2, 1)), np.zeros((2, 2)), np.eye(2))

        a = np.zeros((self.period + 1, 1))
        P_star = np.zeros((self.period + 1, self.period + 1))
        P_inf = np.eye(self.period + 1)

        return DiffuseState(a, P_star, P_inf)

    def _init_observation_matrix(self) -> np.ndarray:
        if self.period is None:
            return np.array([[1, 0]])

        Z = np.zeros((1, self.period + 1))
        Z[:, 0] = 1
        Z[:, 2] = 1
        return Z

    def _init_transition_matrix(self) -> np.ndarray:
        T_linear = np.array([[1, 1], [0, 1]])

        if self.period is None:
            return T_linear

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        return block_diag(T_linear, T_period)

    def _init_transition_error(
        self, mean_noise, trend_noise, period_noise
    ) -> np.ndarray:

        if self.period is None:
            return block_diag(mean_noise, trend_noise)

        return block_diag(
            mean_noise,
            trend_noise,
            period_noise,
            np.zeros((self.period - 2, self.period - 2)),
        )

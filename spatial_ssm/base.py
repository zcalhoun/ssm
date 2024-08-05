"""State space models for temporal and spatiotemporal data.

Base classes for state space models and components for the state space models are 
included in this modele. To define a state space model, you need to first define the 
initial state, then provide the transition and observation models. The base models
may be extended to include more advanced models. Several models are included in this
module that extend the functionality of the base models for typical scenarios.

Typical usage example:

TODO: Add a typical usage example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import repeat
from typing import Tuple, Type

import numpy as np
from scipy.linalg import block_diag

from .utils import Exponential, Kernel


class State:
    """Base class for the state of the state object.

    State space models manipulate *states*, so the input to the state space model
    at each timestep is a state object.

    Attributes:
        mu: The mean of the state.
        covariance: The covariance of the state.
    """

    def __init__(self, mu: np.ndarray, covariance: np.ndarray):
        """Initialize the state object

        Args:
            mu: The mean of the state.
            covariance: The covariance of the state.
        """
        self.mu = mu
        self.covariance = covariance

        # if covariance isn't a list or array, raise error
        if not isinstance(self.covariance, (list, np.ndarray)):
            raise ValueError("Covariance must be a list or numpy array.")

        # check if mu is a list or array, and raise an error
        if not isinstance(self.mu, (list, np.ndarray)):
            raise ValueError("Mean must be a list or numpy array.")

        if len(self.mu) != self.covariance.shape[0]:
            raise ValueError(
                f"Shape of mean vector {self.mu.shape} does not match "
                f"shape of covariance matrix {self.covariance.shape}"
            )

        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("State covariance matrix must be square.")

    def __len__(self) -> int:
        return len(self.mu)

    def copy(
        self, mu: np.ndarray | None = None, covariance: np.ndarray | None = None
    ) -> State:
        """Copy the state and update with new mean and covariance.

        Args:
            mu: The new mean, if provided.
            covariance: The new covariance, if provided.
        """
        new_state = deepcopy(self)
        if mu is not None:
            if mu.shape != self.mu.shape:
                raise ValueError(
                    f"Shape of mean vector {mu.shape} does not match "
                    f"shape of original mean vector {self.mu.shape}"
                )
            new_state.mu = mu
        if covariance is not None:
            if covariance.shape != self.covariance.shape:
                raise ValueError(
                    f"Shape of covariance matrix {covariance.shape} does not match "
                    f"shape of original covariance matrix {self.covariance.shape}"
                )
            new_state.covariance = covariance
        return new_state


class StateSpaceModel(ABC):
    """The base class for state space models.

    This class defines the basic structure of a state space model, to be implemented by
    other modules.
    """

    @abstractmethod
    def predict(self, xi: State) -> State:
        """The predict step of the state-space model.

        This function should predict the next state in the model.

        Args:
            xi: The current state.

        Returns:
            The predicted state.
        """

    @abstractmethod
    def update(self, xi: State, y: np.ndarray) -> State:
        """The update step of the state-space model.

        Args:
            xi: The predicted state
            y: The observation.

        Returns:
            The updated state.
        """

    @abstractmethod
    def filter(self, data) -> Tuple[State]:
        """Filter the data and return the list of states.

        Args:
            data: The data to filter.

        Returns:
            The list of states.
        """

    @abstractmethod
    def smooth(self, data) -> Tuple[State]:
        """Smooth the data

        This function should call the filter method first (i.e., the forwards algorithm)
        then apply the backwards algorithm to smooth the states.
        Args:
            data: The data to smooth.

        Returns:
            The list of smoothed states.
        """


class SpatialState(State):
    def __init__(self, mu, covariance, coords, offset=0) -> None:
        super().__init__(mu, covariance)
        self.coords = coords
        self.offset = offset

    # def interpolate(self, indices, return_H=False):
    #     K_yx = self.kernel(indices, self.coords)
    #     # K_xx = self.kernel(self.coords, self.coords)

    #     if return_H:
    #         H = K_yx @ np.linalg.inv(self.covariance)
    #         mu = H @ self.mu
    #         return mu, H
    #     else:
    #         mu = K_yx @ np.linalg.inv(self.covariance) @ self.mu
    #         return mu

    def fill_shape(self, shape: np.ndarray, kernel: Type[Kernel]) -> np.ndarray:

        full = np.indices(shape).reshape(2, -1).T
        K_yx = kernel(full, self.coords)

        K_xx = kernel(self.coords, self.coords)

        # preds = K_yx @ np.linalg.inv(self.covariance) @ self.mu
        preds = K_yx @ np.linalg.inv(K_xx) @ self.mu[self.offset :]

        return preds.reshape(shape)


class PeriodicRandomWalk(StateSpaceModel):
    """
    This class combines the periodic and random walk models for the states, so that we can
    learn to predict the mean perfectly at each time step, but with the periodic part learned
    so that we can predict the future better.
    """

    def __init__(
        self,
        initial_state: State,
        period: int = 24,
        random_walk_noise: float = 1,
        periodic_noise: float = 1,
        observation_noise: float = 1,
    ) -> None:

        self.initial_state = initial_state
        self.period = period
        self.observation_noise = observation_noise

        self.transition_error = block_diag(
            random_walk_noise, periodic_noise, np.zeros((period - 2, period - 2))
        )
        self.transition_matrix = self._init_transition_matrix()
        self.observation_matrix = self._init_observation_matrix()

    def _init_observation_matrix(self) -> np.ndarray:
        """
        This method initializes the observation matrix.
        """
        Z = np.zeros(self.period)
        Z[0] = 1  # The random walk component
        Z[1] = 1  # The periodic component
        return Z

    def _init_transition_matrix(self) -> np.ndarray:
        """
        This method initializes the transition matrix.
        """
        T_noise = 1

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        return block_diag(T_noise, T_period)

    def predict(self, current_state: Type[State]) -> State:
        mu = self.transition_matrix @ current_state.mu
        P = (
            self.transition_matrix @ current_state.covariance @ self.transition_matrix.T
            + self.transition_error
        )
        return current_state.copy(mu, P)

    def update(self, current_state: Type[State], observation) -> State:
        v = observation - self.observation_matrix @ current_state.mu
        S = (
            self.observation_matrix
            @ current_state.covariance
            @ self.observation_matrix.T
            + self.observation_noise
        )

        if S.ndim == 0:
            K = current_state.covariance @ self.observation_matrix.T / S
        else:
            K = current_state.covariance @ self.observation_matrix.T @ np.linalg.inv(S)

        mu = current_state.mu + K * v

        # A more numerically stable version of the covariance update
        covariance = (
            current_state.covariance
            - K @ self.observation_matrix * current_state.covariance
        )

        return current_state.copy(mu, covariance)

    def smooth(self):
        raise NotImplementedError


class LocalLinearTrend(PeriodicRandomWalk):
    """
    This model extends the periodic random walk by adding in a linear
    trend model which is only updated via the state model.

    The only thing that will change is the definitions of the transition matrix
    and the observation matrix.
    """

    def __init__(
        self,
        initial_state: State,
        period: int = 24,
        random_walk_noise: float = 1,
        linear_noise: float = 1,
        periodic_noise: float = 1,
        observation_noise: float = 1,
    ) -> None:
        super().__init__(initial_state, period)

        self.transition_error = block_diag(
            random_walk_noise,
            linear_noise,
            periodic_noise,
            np.zeros((period - 2, period - 2)),
        )

        self.transition_matrix = self._init_transition_matrix()
        self.observation_matrix = self._init_observation_matrix()

    def _init_transition_matrix(self) -> np.ndarray:
        T_linear = [[1, 1], [0, 1]]

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        return block_diag(T_linear, T_period)

    def _init_observation_matrix(self) -> np.ndarray:
        Z = np.zeros(self.period + 1)
        Z[0] = 1
        Z[2] = 1
        return Z


class Periodic(StateSpaceModel):

    def __init__(
        self,
        initial_state: Type[State],
        period: int = 24,
        transition_error: float = 1,
        measurement_error: float = 1,
        trigonometric: bool = True,
    ):
        """
        This class implements the periodic model.

        Arguments:

        :param initial_state: The initial state of the model.
        :param trigonometric: A boolean flag to indicate whether to use trigonometric functions.
        """
        self.initial_state = initial_state
        self.period = period
        self.trigonometric = trigonometric
        if trigonometric:
            self.transition_error = transition_error * np.eye(period - 1)
            self.measurement_error = measurement_error * np.eye(period - 1)
        else:
            self.transition_error = transition_error
            self.measurement_error = measurement_error
        self.transition_matrix = self._init_transition_matrix()
        self.observation_matrix = self._create_observation_matrix()

    def _create_observation_matrix(self):

        if self.trigonometric:
            z = np.zeros(self.period - 1)
            z[:2] = 1
        else:
            z = np.zeros(self.period - 1)
            z[0] = 1
        return z

    def _init_transition_matrix(self):

        if self.trigonometric:
            T = []
            for i in range(1, self.period // 2):
                C_i = np.matrix(
                    [
                        [
                            np.cos(2 * np.pi * i / self.period),
                            np.sin(2 * np.pi * i / self.period),
                        ],
                        [
                            -np.sin(2 * np.pi * i / self.period),
                            np.cos(2 * np.pi * i / self.period),
                        ],
                    ]
                )
                T.append(C_i)

            T.append([-1])
            T = block_diag(*T)
        else:
            T = np.zeros((self.period - 1, self.period - 1))
            # make first row -1s
            T[0, :] = -1
            for i in range(1, self.period - 1):
                T[i, i - 1] = 1

        return T

    def predict(self, current_state: Type[State]) -> State:
        mu = self.transition_matrix @ current_state.mu
        P = (
            self.transition_matrix @ current_state.covariance @ self.transition_matrix.T
            + self.transition_error
        )
        return current_state.copy(mu, P)

    def update(self, current_state: Type[State], observation) -> State:
        v = observation - self.observation_matrix @ current_state.mu
        S = (
            self.observation_matrix
            @ current_state.covariance
            @ self.observation_matrix.T
            + self.measurement_error
        )

        if self.trigonometric:
            K = current_state.covariance @ self.observation_matrix.T @ np.linalg.inv(S)
        else:
            K = current_state.covariance @ self.observation_matrix.T / S

        mu = current_state.mu + K * v
        covariance = (
            current_state.covariance
            - K @ self.observation_matrix * current_state.covariance
        )

        return current_state.copy(mu, covariance)

    def smooth(self, data) -> Tuple[State]:
        raise NotImplementedError


class RandomWalk(StateSpaceModel):
    """
    This class implements the basic random walk model.
    """

    def __init__(
        self,
        initial_state: Type[State],
        transition_error: float = 1,
        measurement_error: float = 1,
    ):
        self.initial_state = initial_state
        self.transition_error = transition_error
        self.measurement_error = measurement_error

    def predict(self, current_state: Type[State]) -> State:
        mu = current_state.mu
        P = current_state.covariance + self.transition_error
        return current_state.copy(mu, P)

    def update(self, current_state: Type[State], observation) -> State:
        mu = current_state.mu + current_state.covariance / (
            current_state.covariance + self.measurement_error
        ) * (observation - current_state.mu)

        P = (
            current_state.covariance
            - current_state.covariance
            / (current_state.covariance + self.measurement_error)
            * current_state.covariance
        )

        return current_state.copy(mu, P)

    def smooth(self, data):
        states = self.filter(data)
        smoothed = [states[-1]]
        for s in reversed(states[:-1]):
            s_plus = self.predict(s)
            G = s.covariance / (s.covariance + self.transition_error)
            mu_plus = s.mu + G * (smoothed[-1].mu - s_plus.mu)
            covariance_plus = (
                s.covariance - G * s.covariance * G * smoothed[-1].covariance
            )
            smoothed_state = s.copy(mu_plus, covariance_plus)
            smoothed.append(smoothed_state)

        return smoothed[::-1]


class SpatialRandomWalk(StateSpaceModel):
    """This class implements the spatial random walk model"""

    def __init__(
        self,
        initial_state: Type[State],
        alpha: int = 0.1,
        spatial_kernel: Type[Kernel] = Exponential(),
        measurement_error: float = 0.1,
    ) -> None:
        """
        Arguments:
            initial_state: The initial state of the model.
            transition: The transition model.
            observation: The observation model.
        """
        self.initial_state = initial_state
        self.alpha = alpha
        self.spatial_kernel = spatial_kernel
        self.measurement_error = measurement_error

    def predict(self, current_state: Type[SpatialState]) -> State:
        m = self.alpha * current_state.mu
        P = self.alpha**2 * current_state.covariance
        P += self.spatial_kernel(current_state.coords, current_state.coords) * (
            1 - self.alpha**2
        )

        return current_state.copy(m, P)

    def update(self, current_state, observation) -> State:
        indices = np.stack(np.where(~observation.mask), axis=-1)
        H = self.spatial_kernel(indices, current_state.coords)
        K_xx = self.spatial_kernel(current_state.coords, current_state.coords)
        K_yy = self.spatial_kernel(indices, indices)
        obs = observation.data[~observation.mask]
        R = (
            self.measurement_error * np.eye(len(obs))
            + K_yy
            - H @ np.linalg.inv(K_xx) @ H.T
        )
        H = H @ np.linalg.inv(K_xx)

        pred = H @ current_state.mu
        v = obs - pred
        # R = self.measurement_error * np.eye(len(obs)) + H
        S = H @ current_state.covariance @ H.T + R
        K = current_state.covariance @ H.T @ np.linalg.inv(S)

        mu = current_state.mu + K @ v
        covariance = current_state.covariance - K @ H @ current_state.covariance

        return current_state.copy(mu, covariance)

    def filter(self, data) -> Tuple[State]:
        current_state = self.initial_state
        states = [current_state]
        for y in data:
            current_state = self.predict(current_state)
            current_state = self.update(current_state, y)
            states.append(current_state)
        return states

    def smooth(self, data) -> Tuple[State]:
        states = self.filter(data)
        smoothed = [states[-1]]
        for s in reversed(states[:-1]):
            s_plus = self.predict(s)
            G = self.alpha * s.covariance @ np.linalg.inv(s_plus.covariance)
            mu_plus = s.mu + G @ (smoothed[-1].mu - s_plus.mu)
            covariance_plus = (
                s.covariance + G @ (smoothed[-1].covariance - s_plus.covariance) @ G.T
            )
            smoothed_state = s.copy(mu_plus, covariance_plus)
            smoothed.append(smoothed_state)

        return smoothed[::-1]


class SpatialLinearTrend(StateSpaceModel):

    def __init__(
        self,
        initial_state: Type[SpatialState],
        alpha: float = 0.1,
        spatial_kernel: Type[Kernel] = Exponential(),
        observation_noise: float = 0.1,
        random_walk_noise: float = 1,
        linear_noise: float = 1,
        periodic_noise: float = 1,
        period: int = 24,
        obs_coords: np.ndarray = None,
    ) -> None:
        self.initial_state = initial_state
        self.alpha = alpha
        self.spatial_kernel = spatial_kernel
        self.period = period
        self.num_points = len(initial_state.coords)
        self.obs_coords = obs_coords

        self.transition_matrix = self._init_transition_matrix()

        self.transition_error = block_diag(
            random_walk_noise,
            linear_noise,
            periodic_noise,
            np.zeros((period - 2, period - 2)),
            (1 - alpha**2) * spatial_kernel(initial_state.coords, initial_state.coords),
        )

        self.observation_error = observation_noise * np.eye(len(obs_coords))

        self.observation_matrix = self._init_observation_matrix()

    def predict(self, current_state: Type[SpatialState]) -> State:
        m = self.transition_matrix @ current_state.mu
        P = (
            self.transition_matrix @ current_state.covariance @ self.transition_matrix.T
            + self.transition_error
        )
        return current_state.copy(m, P)

    def update(self, current_state: Type[SpatialState], observation) -> State:
        v = observation - self.observation_matrix @ current_state.mu

        # Replace nans with zeros
        v[np.isnan(v)] = 0

        S = (
            self.observation_matrix
            @ current_state.covariance
            @ self.observation_matrix.T
            + self.observation_error
        )

        K = current_state.covariance @ self.observation_matrix.T @ np.linalg.inv(S)

        mu = current_state.mu + K @ v

        covariance = (
            current_state.covariance
            - K @ self.observation_matrix @ current_state.covariance
        )
        return current_state.copy(mu, covariance)

    def _init_observation_matrix(self) -> np.ndarray:
        """
        This function generates the observation matrix for the spatial
        linear trend.

        Returns:
            Z: The observation matrix.

        """

        z_temporal = np.zeros(self.period + 1)
        z_temporal[0] = 1
        z_temporal[2] = 1

        z_spatial = self.spatial_kernel(self.obs_coords, self.initial_state.coords)
        z_inv = np.linalg.inv(
            self.spatial_kernel(self.initial_state.coords, self.initial_state.coords)
        )

        Z = np.hstack(
            (
                np.ones((len(self.obs_coords), 1)) @ z_temporal[np.newaxis, :],
                z_spatial @ z_inv,
            )
        )
        return Z

    def _init_transition_matrix(self) -> np.ndarray:
        """
        This function generates the transition matrix for the spatial
        linear trend.

        The transition matrix is defined by:
        1. The linear trend component.
        2. The periodic component.
        3. The spatial component.

        Returns:
            T: The transition matrix.

        """

        T_linear = [[1, 1], [0, 1]]

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        T_spatial = self.alpha * np.eye(self.num_points)

        return block_diag(T_linear, T_period, T_spatial)

    def smooth(self, data) -> Tuple[State]:
        raise NotImplementedError


class SpatialLinearTrendWithCovariates(SpatialLinearTrend):

    def __init__(
        self,
        initial_state: Type[SpatialState],
        alpha: float = 0.1,
        spatial_kernel: Type[Kernel] = Exponential(),
        observation_noise: float = 0.1,
        random_walk_noise: float = 1,
        linear_noise: float = 1,
        periodic_noise: float = 1,
        period: int = 24,
        obs_coords: np.ndarray = None,
        covariates: np.ndarray = None,
        covariate_noise: float = 1,
    ) -> None:
        self.covariates = covariates
        super().__init__(
            initial_state,
            alpha,
            spatial_kernel,
            observation_noise,
            random_walk_noise,
            linear_noise,
            periodic_noise,
            period,
            obs_coords,
        )

        self.transition_error = block_diag(
            random_walk_noise,
            linear_noise,
            periodic_noise,
            np.zeros((period - 2, period - 2)),
            covariate_noise * np.eye(covariates.shape[1]),
            (1 - alpha**2) * spatial_kernel(initial_state.coords, initial_state.coords),
        )

    def _init_observation_matrix(self) -> np.ndarray:
        """
        This function generates the observation matrix for the spatial
        linear trend with covariates.

        Returns:
            Z: The observation matrix.

        """

        z_temporal = np.zeros(self.period + 1)
        z_temporal[0] = 1
        z_temporal[2] = 1

        z_spatial = self.spatial_kernel(self.obs_coords, self.initial_state.coords)
        z_inv = np.linalg.inv(
            self.spatial_kernel(self.initial_state.coords, self.initial_state.coords)
        )

        Z = np.hstack(
            (
                np.ones((len(self.obs_coords), 1)) @ z_temporal[np.newaxis, :],
                self.covariates,
                z_spatial @ z_inv,
            )
        )
        return Z

    def _init_transition_matrix(self) -> np.ndarray:
        """
        This function generates the transition matrix for the spatial
        linear trend.

        The transition matrix is defined by:
        1. The linear trend component.
        2. The periodic component.
        3. The spatial component.

        Returns:
            T: The transition matrix.

        """

        T_linear = [[1, 1], [0, 1]]

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        T_covariates = np.eye(self.covariates.shape[1])
        T_spatial = self.alpha * np.eye(self.num_points)

        return block_diag(T_linear, T_period, T_covariates, T_spatial)


class SpatialLinearTrendWithPeriodicCovariates(SpatialLinearTrend):

    def __init__(
        self,
        initial_state: Type[SpatialState],
        alpha: float = 0.1,
        spatial_kernel: Type[Kernel] = Exponential(),
        observation_noise: float = 0.1,
        random_walk_noise: float = 1,
        linear_noise: float = 1,
        periodic_noise: float = 1,
        period: int = 24,
        obs_coords: np.ndarray = None,
        covariates: np.ndarray = None,
        covariate_noise: float = 1,
    ) -> None:

        self.num_covariates = covariates.shape[1]
        self.covariates = covariates
        super().__init__(
            initial_state,
            alpha,
            spatial_kernel,
            observation_noise,
            random_walk_noise,
            linear_noise,
            periodic_noise,
            period,
            obs_coords,
        )

        self.transition_error = block_diag(
            random_walk_noise,
            linear_noise,
            create_periodic_error(period, periodic_noise),
            block_diag(
                *repeat(
                    create_periodic_error(period, covariate_noise), self.num_covariates
                )
            ),
            (1 - alpha**2) * spatial_kernel(initial_state.coords, initial_state.coords),
        )

    def _init_observation_matrix(self) -> np.ndarray:
        """
        This function generates the observation matrix for the spatial
        linear trend with covariates.

        Returns:
            Z: The observation matrix.

        """

        z_temporal = np.zeros(2 + self.period - 1)
        z_temporal[0] = 1
        z_temporal[2] = 1

        z_covariates = np.zeros(
            (len(self.obs_coords), self.num_covariates * (self.period - 1))
        )
        z_covariates[:, :: self.period - 1] = self.covariates

        z_spatial = self.spatial_kernel(self.obs_coords, self.initial_state.coords)
        z_inv = np.linalg.inv(
            self.spatial_kernel(self.initial_state.coords, self.initial_state.coords)
        )

        Z = np.hstack(
            (
                np.ones((len(self.obs_coords), 1)) @ z_temporal[np.newaxis, :],
                z_covariates,
                z_spatial @ z_inv,
            )
        )
        return Z

    def _init_transition_matrix(self) -> np.ndarray:
        """
        This function generates the transition matrix for the spatial
        linear trend.

        The transition matrix is defined by:
        1. The linear trend component.
        2. The periodic component.
        3. The spatial component.

        Returns:
            T: The transition matrix.

        """

        T_linear = [[1, 1], [0, 1]]

        T_period = np.zeros((self.period - 1, self.period - 1))
        T_period[0, :] = -1
        for i in range(1, self.period - 1):
            T_period[i, i - 1] = 1

        # T_covariates = np.eye(self.covariates.shape[1])
        T_spatial = self.alpha * np.eye(self.num_points)

        return block_diag(
            T_linear, block_diag(*repeat(T_period, self.num_covariates + 1)), T_spatial
        )


def create_periodic_error(period, noise):
    """
    This function generates the periodic error matrix

    Arguments:
    n: The size of the matrix.
    val: The value to fill the matrix with.

    Example:
    >>> create_periodic_error(5, 0.1)

    Returns:
    >>> array([[0.1, 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0.]])
    """
    m = np.zeros((period - 1, period - 1))
    m[0, 0] = noise
    return m

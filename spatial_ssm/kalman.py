"""Linear Gaussian State Space Models

This module contains the Kalman Filter class, which contains the implementation of the
Kalman Filter algorithm for linear Gaussian State Space Models. This also contains functions 
that automate the initilization of the Kalman Filter for specific use cases.

Classes:
    LinearGaussianSSM: A class for linear Gaussian State Space Models.

Functions:

"""

from typing import List, Tuple

import numpy as np

from .base import State, StateSpaceModel, DiffuseState


class LinearGaussianSSM(StateSpaceModel):
    """The Linear Gaussian State Space Model class.

    Attributes:
        A: the transition matrix
        H: the observation matrix
        Q: the transition noise matrix
        R: the observation noise matrix
        x0: the initial state
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        transition_noise: np.ndarray,
        observation_noise: np.ndarray,
        initial_state: State | None = None,
    ):  # pylint: disable=too-many-arguments
        """Initialize the Linear Gaussian State Space Model.

        Args:
            transition_matrix: The transition matrix.
            observation_matrix: The observation matrix.
            transition_noise: The transition noise matrix.
            observation_noise: The observation noise matrix.
            initial_state: The initial state of the model. Defaults to None.

        Raises:
            ValueError: If the dimensions of the matrices are not compatible.
        """
        self.A = transition_matrix
        self.H = observation_matrix
        self.Q = transition_noise
        self.R = observation_noise

        if initial_state is None:
            self.x0 = State(np.zeros(self.A.shape[0]), np.eye(self.A.shape[0]))
        else:
            self.x0 = initial_state

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(
                f"Transition matrix must be square. Got shape {self.A.shape}."
            )

        if self.A.shape[0] != len(self.x0.mu):
            raise ValueError(
                f"State size {len(self.x0.mu)} must match the dimension "
                + f"{self.A.shape[0]} of the transition matrix."
            )

        if self.Q.shape[0] != len(self.x0.mu):
            raise ValueError(
                f"State size {len(self.x0.mu)} must match the row dimension "
                + f"{self.Q.shape[0]} of the transition noise matrix."
            )

        if self.H.shape[0] != self.R.shape[0]:
            raise ValueError(
                f"Observation column size {self.H.shape[1]} does not match "
                + f"observation noise matrix row size {self.R.shape[0]}."
            )

        if self.H.shape[1] != len(self.x0.mu):
            raise ValueError(
                f"State dimension {len(self.x0.mu)} must match "
                + f"the column size {self.H.shape[1]} of the observation matrix."
            )

    def predict(self, xi: State) -> State:
        mu = self.A @ xi.mu
        sigma = self.A @ xi.covariance @ self.A.T + self.Q

        return xi.copy(mu, sigma)

    def update(self, xi: State, y: np.ndarray) -> State:
        v = y - self.H @ xi.mu
        # Handle missing values
        v[np.isnan(y)] = 0
        s = self.H @ xi.covariance @ self.H.T + self.R
        k = xi.covariance @ self.H.T @ np.linalg.inv(s)

        mu = xi.mu + k @ v
        sigma = xi.covariance - k @ self.H @ xi.covariance

        return xi.copy(mu, sigma)

    def filter(self, data: np.ndarray) -> List[State]:
        current_state = self.x0
        xi = [self.x0]
        for y in data:
            if isinstance(current_state, DiffuseState):
                current_state = self._diffuse(current_state, y)
            else:
                current_state = self.predict(current_state)
                current_state = self.update(current_state, y)
            xi.append(current_state)
        return xi

    def _diffuse(self, xi: DiffuseState, y: np.ndarray) -> State:
        """This function runs the diffuse Kalman filter."""
        F_inf = self.H @ xi.diffuse_covariance @ self.H.T
        F_star = self.H @ xi.covariance @ self.H.T + self.R
        M_inf = xi.diffuse_covariance @ self.H.T
        M_star = xi.covariance @ self.H.T

        # check if F_inf == 0
        if np.all(F_inf == 0):
            K_0 = self.A @ M_star @ np.linalg.inv(F_star)
            L_0 = self.A - K_0 @ self.H
            P_inf = self.A @ xi.diffuse_covariance @ self.A.T
            P_star = self.A @ xi.covariance @ L_0.T + self.Q
            v_0 = y - self.H @ xi.mu
            a_0 = self.A @ xi.mu + K_0 @ v_0

            # for numerical reasons, we check if values are small then return
            # to regular states
            if np.all(P_inf < 1e-6):
                return State(a_0, P_star)

            return xi.copy(a_0, P_star, P_inf)

        F_1 = np.linalg.inv(F_inf)
        F_2 = -F_1 @ F_star @ F_1
        K_0 = self.A @ M_inf @ F_1
        K_1 = self.A @ M_star @ F_1 + self.A @ M_inf @ F_2
        L_0 = self.A - K_0 @ self.H
        L_1 = -K_1 @ self.H

        v_0 = y - self.H @ xi.mu
        a_0 = self.A @ xi.mu + K_0 @ v_0

        P_inf = self.A @ xi.diffuse_covariance @ L_0.T
        P_star = (
            self.A @ xi.diffuse_covariance @ L_1.T
            + self.A @ xi.covariance @ L_0.T
            + self.Q
        )
        # Once the numbers get small, we cut out
        if np.all(P_inf < 1e-6):
            return State(a_0, P_star)

        return xi.copy(a_0, P_star, P_inf)

    def smooth(self, data: np.ndarray) -> List[State]:
        states = self.filter(data)
        smoothed = [states[-1]]

        for s in reversed(states[:-1]):
            s_plus = self.predict(s)
            G = (  # pylint: disable=invalid-name
                s.covariance @ self.A.T @ np.linalg.inv(s_plus.covariance)
            )

            mu = s.mu + G @ (smoothed[-1].mu - s_plus.mu)
            sigma = (
                s.covariance + G @ (smoothed[-1].covariance - s_plus.covariance) @ G.T
            )

            smoothed.append(s.copy(mu, sigma))

        return smoothed[::-1]


def random_walk(
    x0: State | None = None, q: float | None = 1, r: float | None = 1
) -> LinearGaussianSSM:
    """Create a random walk model.

    Args:
        y: The data.
        x0: The initial state.
        q: The transition noise. Defaults to 1.
        r: The observation noise. Defaults to 1.

    Returns:
        The random walk model.
    """
    A = np.array([[1]])  # pylint: disable=invalid-name
    H = np.array([[1]])  # pylint: disable=invalid-name
    Q = np.array([q])  # pylint: disable=invalid-name
    R = np.array([r])  # pylint: disable=invalid-name

    if x0 is None:
        x0 = State(np.array([0]), np.array([[1]]))

    return LinearGaussianSSM(A, H, Q, R, x0)

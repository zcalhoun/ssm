"""Classes for extended Kalman filtering and smoothing."""

from __future__ import annotations
from abc import ABC

import numpy as np

from .base import State, StateSpaceModel


class ExtendedFirstOrder(StateSpaceModel):
    """First-order extended Kalman filter and smoother.

    This class implements the extended Kalman filter and smoother for a
    first-order state space model. The model is defined by the following
    equations:

        x_{t+1} = f(x_t, q_t)
        y_t = h(x_t, r_t)

    where x_t is the state vector, y_t is the observation vector, f is the state
    transition function, h is the observation function, q_t is the process noise,
    and r_t is the observation noise.

    Attributes:
        f (callable): State transition function.
        h (callable): Observation function.
        F (callable): State transition Jacobian.
        H (callable): Observation Jacobian.
        Q (callable): Process noise covariance.
        R (callable): Observation noise covariance.
    """

    def __init__(
        self,
        transition_function: FirstOrderFunction,
        observation_function: FirstOrderFunction,
        process_noise_covariance: np.ndarray,
        observation_noise_covariance: np.ndarray,
        initial_state: State | None = None,
    ):
        """Initialize the first-order extended Kalman filter and smoother.

        Args:
            transition_function: The state transition function.
            observation_function: The observation function.
            process_noise_covariance: The process noise covariance.
            observation_noise_covariance: The observation noise covariance.
        """
        self.f = transition_function
        self.h = observation_function
        self.Q = process_noise_covariance
        self.R = observation_noise_covariance

        if initial_state is None:
            self.x0 = State(np.zeros(self.f(None).shape), np.eye(self.f(None).shape[0]))
        else:
            self.x0 = initial_state

    def predict(self, xi: State) -> State:
        """Predict the next state given the current state.

        Args:
            state: The current state.

        Returns:
            The predicted state.
        """
        x = xi.mu
        P = xi.covariance

        x_pred = self.f(x)
        F_x = self.f.jac_x(x)
        F_q = self.f.jac_noise(x)
        P_pred = F_x @ P @ F_x.T + F_q @ self.Q @ F_q.T

        return State(x_pred, P_pred)

    def update(self, xi: State, y: np.ndarray):
        """Update the state given the observation.

        Args:
            state: The current state.
            observation: The observation.

        Returns:
            The updated state.
        """
        x = xi.mu
        P = xi.covariance

        y_pred = self.h(x)
        H_x = self.h.jac_x(x, self.R)
        H_r = self.h.jac_noise(x, self.R)
        v = y - y_pred
        v[np.isnan(v)] = 0
        S = H_x @ P @ H_x.T + H_r @ self.R @ H_r.T
        K = P @ H_x.T @ np.linalg.inv(S)

        x_upd = x + K @ v
        P_upd = P - K @ S @ K.T

        return State(x_upd, P_upd)

    def filter(self, data: np.ndarray) -> list[State]:
        """Filter the data.

        Args:
            data: The data.

        Returns:
            The filtered states.
        """
        current_state = self.x0
        xi = [self.x0]
        for y in data:
            current_state = self.predict(current_state)
            current_state = self.update(current_state, y)
            xi.append(current_state)
        return xi

    def smooth(self, data: np.ndarray) -> list[State]:
        raise NotImplementedError


class FirstOrderFunction(ABC):
    """Defines the base functionality needed for a function with a
    first-order derivative (i.e., a Jacobian)."""

    def __call__(self, x: np.ndarray | None) -> np.ndarray:
        """Evaluate the function at the given state.

        Args:
            x: the state value. If none, then the function is treated as a
                constant.

        Returns:
            The function value.
        """
        raise NotImplementedError

    def jac_x(self, x: np.ndarray | None, noise: np.ndarray | None) -> np.ndarray:
        """Evaluate the Jacobian with respect to the state at the given state.

        Args:
            x: the state value. If none, then treated as a constant.

        Returns:
            The Jacobian with respect to the state.

        """
        raise NotImplementedError

    def jac_noise(self, x: np.ndarray | None, noise: np.ndarray | None) -> np.ndarray:
        """Evaluate the Jacobian with respect to the noise at the given state.

        Args:
            noise: the noise value.

        Returns:
            The Jacobian with respect to the noise.
        """
        raise NotImplementedError

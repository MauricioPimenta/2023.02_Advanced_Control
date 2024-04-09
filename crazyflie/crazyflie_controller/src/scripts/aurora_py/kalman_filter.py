import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, D, Rw, Rv, initial_state, initial_error_covariance, Ts):
        self.A = (np.eye(A.shape[0]) + np.array(A)*Ts)
        self.B = np.array(B)*Ts

        self.C = np.array(C)
        self.D = np.array(D)
        self.Ts = Ts
        self.Rw = np.array(Rw)
        self.Rv = np.array(Rv)
        self.state_estimate = np.array(initial_state)
        self.error_covariance = np.array(initial_error_covariance)

    def update(self, measurement, control_input):
        measurement = np.array(measurement)
        control_input = np.array(control_input)

        # Innovation
        innovation = measurement - self.C @ self.state_estimate - self.D @ control_input
        innovation_covariance = self.C @ self.error_covariance @ self.C.T + self.Rv

        # Correction
        kalman_gain = self.error_covariance @ self.C.T @ np.linalg.pinv(innovation_covariance)
        self.state_estimate = self.state_estimate + kalman_gain @ innovation
        self.error_covariance = (np.eye(self.A.shape[0]) - kalman_gain @ self.C) @ self.error_covariance

        # Prediction
        self.old_state = self.state_estimate
        self.state_estimate = self.A @ self.state_estimate + self.B @ control_input
        self.error_covariance = self.A @ self.error_covariance @ self.A.T + self.Rw

        self.state_estimate[2] = self.old_state[2] + self.Ts * self.old_state[5]
        self.state_estimate[5] = self.old_state[5] + self.Ts * (1/0.035 * control_input[2] - 9.81)

        return self.state_estimate, kalman_gain
"""Unscented Kalman Filter for the same tracking problem as kalman_track.run_ekf.

Instead of linearizing with Jacobians, the UKF pushes 2n+1 "sigma points" through
the real nonlinear dynamics and measurement functions and rebuilds the mean and
covariance from where they land (Wan & van der Merwe, 2000).
"""
import numpy as np

from kalman_track import DT, initial_state, measure, process_noise, propagate, residual

ALPHA, BETA, KAPPA = 1e-3, 2.0, 0.0   # standard scaled-sigma-point choices for Gaussian noise


def sigma_weights(n):
    lam = ALPHA**2 * (n + KAPPA) - n
    wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - ALPHA**2 + BETA)
    return lam, wm, wc


def sigma_points(x, P, lam):
    n = len(x)
    L = np.linalg.cholesky((n + lam) * P)
    return np.vstack([x, x + L.T, x - L.T])


def mean_measurement(Z, wm):
    """Weighted mean of measurement sigma points; azimuth is averaged on the circle."""
    z = wm @ Z
    z[1] = np.arctan2(wm @ np.sin(Z[:, 1]), wm @ np.cos(Z[:, 1])) % (2 * np.pi)
    return z


def run_ukf(sites, rots, meas, noise, dt=DT):
    R = np.diag(noise**2)
    Q = process_noise(dt)
    x, P = initial_state(sites, rots, meas, noise, dt)
    lam, wm, wc = sigma_weights(len(x))
    est, covs = [np.full(6, np.nan), x.copy()], [np.full((6, 6), np.nan), P.copy()]
    for k in range(2, len(meas)):
        # Predict: propagate every sigma point through the orbit dynamics
        X = np.array([propagate(s, dt) for s in sigma_points(x, P, lam)])
        x = wm @ X
        dX = X - x
        P = dX.T @ (wc[:, None] * dX) + Q
        # Update: redraw sigma points, push them through the radar model
        X = sigma_points(x, P, lam)
        Z = np.array([measure(s[:3], sites[k], rots[k]) for s in X])
        z = mean_measurement(Z, wm)
        dZ = np.array([residual(zi, z) for zi in Z])
        S = dZ.T @ (wc[:, None] * dZ) + R
        C = (X - x).T @ (wc[:, None] * dZ)
        K = C @ np.linalg.inv(S)
        x = x + K @ residual(meas[k], z)
        P = P - K @ S @ K.T
        P = (P + P.T) / 2
        est.append(x.copy()); covs.append(P.copy())
    return np.array(est), np.array(covs)

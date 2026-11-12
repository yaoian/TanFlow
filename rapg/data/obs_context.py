"""Observation context construction for trajectory velocity conditioning.

Constructs context channels [v_obs_x, v_obs_y, m_obs] representing observed
interval velocities and validity masks over the trajectory window. Missing
gap coordinates are excluded prior to differencing to ensure no label leakage.
"""

from __future__ import annotations

import torch

OBS_CONTEXT_ARMS = ("off", "zero", "past", "future", "bi")
MISSING_POINTS = (10, 11, 12, 13, 14)
_PAST_INTERVALS = (0, 9)
_FUTURE_INTERVALS = (15, 24)


def validate_arm(arm: str) -> str:
    if arm not in OBS_CONTEXT_ARMS:
        raise ValueError(f"unknown obs_context arm {arm!r}; expected one of {list(OBS_CONTEXT_ARMS)}")
    return arm


def observation_mask(points: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """Return point observation mask q: 1.0 for observed points, 0.0 for missing gap."""
    q = torch.ones(points, dtype=dtype, device=device)
    q[list(MISSING_POINTS)] = 0.0
    return q


def interval_mask(points: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """Return interval observation mask m_obs: 1.0 where both endpoints are observed."""
    q = observation_mask(points, dtype=dtype, device=device)
    return q[:-1] * q[1:]


def build_obs_context(trajectory: torch.Tensor, dt: torch.Tensor, arm: str) -> torch.Tensor:
    """Build observation context tensor X_obs of shape [B, 3, 24]."""
    validate_arm(arm)
    traj = trajectory.to(torch.float64)
    durations = dt.to(device=traj.device, dtype=torch.float64)
    points = traj.shape[-1]
    q = observation_mask(points, device=traj.device)
    m = interval_mask(points, device=traj.device).expand(traj.shape[0], -1)

    # Zero unobserved positions before differencing to prevent NaN propagation
    observed = torch.where(q[None, None, :].bool(), traj[:, :2], torch.zeros((), dtype=torch.float64))
    v = (observed[:, :, 1:] - observed[:, :, :-1]) / durations[:, None]
    v = v * m[:, None]

    if arm == "zero":
        v = torch.zeros_like(v)
        m = torch.zeros_like(m)
    elif arm in ("past", "future"):
        start, stop = _PAST_INTERVALS if arm == "past" else _FUTURE_INTERVALS
        keep = torch.zeros_like(m)
        keep[:, start:stop] = 1.0
        v = v * keep[:, None]
        m = m * keep

    return torch.cat((v, m[:, None]), 1)

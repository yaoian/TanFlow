"""Standalone V_K: observed boundary kinematics followed by constant closure.
The operation order, native-coordinate conversion and per-window NumPy arithmetic
are retained from the frozen kinematic-only implementation.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Iterator
import numpy as np
import torch

def build_kinematic_prior_velocity(trajectory, mask, coord_mean, coord_std):
    values = trajectory.to(torch.float64)
    mean = torch.as_tensor(coord_mean, dtype=torch.float64, device=values.device)
    std = torch.as_tensor(coord_std, dtype=torch.float64, device=values.device)
    mask = mask.to(device=values.device, dtype=torch.float64)
    return torch.stack([_window_velocity(row, mask, mean, std) for row in values]).contiguous()

def _window_velocity(trajectory, mask, mean, std):
    _validate_input(trajectory, mask, mean, std)
    missing = mask > .1
    time_raw = ((trajectory[2] - trajectory[2, 0]) * std[2]).detach().cpu().numpy()
    if np.any(np.diff(time_raw) <= 1e-6):
        raise ValueError("time must be strictly increasing")
    if not torch.isfinite(trajectory[:2, ~missing]).all():
        raise ValueError("observed coordinates must be finite")
    target = torch.zeros((2, trajectory.shape[-1]-1), dtype=trajectory.dtype, device=trajectory.device)
    correction = torch.zeros_like(target)
    observed = torch.nonzero(~missing, as_tuple=False).flatten()
    for start, end in _segments(missing):
        left, right = start-1, end
        if left < 0 or right >= trajectory.shape[-1]:
            raise ValueError("A missing run must have two observed anchors")
        li = observed[observed <= left][-4:]
        ri = observed[observed >= right][:4]
        lxy = _raw_xy(trajectory[:2, li].T, mean[:2], std[:2])
        rxy = _raw_xy(trajectory[:2, ri].T, mean[:2], std[:2])
        times = time_raw[left:right+1]
        dt = np.diff(times)
        total = float(times[-1]-times[0])
        bridge = (rxy[0]-lxy[-1])/total
        vl = _context_velocity(lxy, time_raw[li.detach().cpu().numpy()], True, 3, bridge, 1e-6)
        vr = _context_velocity(rxy, time_raw[ri.detach().cpu().numpy()], False, 3, bridge, 1e-6)
        ratio = (.5*(times[:-1]+times[1:])-times[0])/total
        inertia = (1-ratio[:,None])*vl[None,:]+ratio[:,None]*vr[None,:]
        base = .5*bridge[None,:]+.5*inertia
        base = np.broadcast_to(base, (len(dt),2)).copy()
        _, bias = _close_and_integrate(lxy[-1], rxy[0], dt, base)
        scale = std[2].reshape(1,1)/std[:2].reshape(1,2)
        target[:,left:right] = (torch.as_tensor(base, dtype=torch.float64, device=trajectory.device)*scale).T
        correction[:,left:right] = (torch.as_tensor(bias, dtype=torch.float64, device=trajectory.device)*scale).T
    return (target+correction).T

def _close_and_integrate(left: np.ndarray, right: np.ndarray, dt: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hard endpoint closure only; no temporal smoothness operation."""
    total = float(dt.sum())
    bias_vector = ((right - left) - (v * dt[:, None]).sum(axis=0)) / total
    bias = np.repeat(bias_vector[None, :], len(dt), axis=0)
    points = left[None, :] + np.cumsum((v + bias) * dt[:, None], axis=0)
    if not np.allclose(points[-1], right, atol=1e-8, rtol=0):
        raise RuntimeError("Kinematic endpoint closure failed")
    return points[:-1], bias


def _context_velocity(xy: np.ndarray, times: Sequence[float], use_last: bool, window: int, fallback: np.ndarray, dt_min: float) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if len(xy) < 2:
        return fallback.copy()
    n = min(len(xy), window + 1)
    x, t = (xy[-n:], times[-n:]) if use_last else (xy[:n], times[:n])
    elapsed = float(t[-1] - t[0])
    if elapsed <= dt_min:
        return fallback.copy()
    v = (x[-1] - x[0]) / elapsed
    return v if np.all(np.isfinite(v)) else fallback.copy()


def _segments(mask: torch.Tensor) -> Iterator[Tuple[int, int]]:
    start = None
    for i in range(len(mask)):
        if bool(mask[i]):
            start = i if start is None else start
        elif start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(mask)


def _raw_xy(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    return (x.to(torch.float64) * std + mean).detach().cpu().numpy()


def _validate_input(trajectory: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    if not torch.is_tensor(trajectory) or trajectory.ndim != 2 or trajectory.shape[0] != 3 or trajectory.shape[1] < 2:
        raise ValueError("trajectory must be torch.Tensor [3,L>=2]")
    if not torch.is_tensor(mask) or mask.ndim != 1 or len(mask) != trajectory.shape[1]:
        raise ValueError("mask must be [L]")
    mean = torch.as_tensor(mean, dtype=torch.float64, device=trajectory.device).reshape(-1)
    std = torch.as_tensor(std, dtype=torch.float64, device=trajectory.device).reshape(-1)
    if len(mean) < 3 or len(std) < 3 or not torch.all(torch.isfinite(mean[:3])) or not torch.all(torch.isfinite(std[:3])) or torch.any(std[:3] <= 0):
        raise ValueError("coord_mean/std must be finite [x,y,time] with std > 0")
    if not torch.all(torch.isfinite(trajectory[2])):
        raise ValueError("time row must be finite")


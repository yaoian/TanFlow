"""Tangent-space Schur source covariance estimation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
EPS_VAR = 1e-6

@dataclass(frozen=True)
class Calibrator:
    alpha: float
    sigma_floor: float
    sigma_unknown: float
    isotropic_variance: float
    eps_var: float = EPS_VAR

    def __post_init__(self) -> None:
        if self.eps_var != EPS_VAR or self.alpha <= 0 or self.sigma_unknown <= 0:
            raise ValueError("calibrator alpha/unknown must be positive and eps_var fixed")
        if self.sigma_floor * self.sigma_floor < EPS_VAR or self.isotropic_variance < EPS_VAR:
            raise ValueError("calibrator variances violate fixed lower bound")
        if not all(math.isfinite(value) for value in (self.alpha, self.sigma_floor, self.sigma_unknown, self.isotropic_variance)):
            raise ValueError("calibrator parameters must be finite")

    def asdict(self) -> dict[str, float]:
        return asdict(self)


def evidence(n_eff: torch.Tensor, distance_ratio: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    if n_eff.shape != distance_ratio.shape or available.shape != n_eff.shape:
        raise ValueError("evidence tensors must share [B,N] shape")
    if not torch.isfinite(n_eff).all() or torch.isnan(distance_ratio).any():
        raise ValueError("n_eff must be finite and distance_ratio cannot be NaN")
    if not torch.all((available == 0) | (available == 1)) or torch.any(n_eff < 0):
        raise ValueError("available must be binary and n_eff nonnegative")
    raw = n_eff.ge(4).to(n_eff.dtype) * (n_eff / 16).clamp(max=1) * torch.exp(-.5 * distance_ratio.square())
    return torch.where(available.bool(), raw, torch.zeros_like(raw))


def isotropic_mle(residual: torch.Tensor, active: torch.Tensor, dt: torch.Tensor | None = None) -> float:
    _validate_calibration_shapes(residual, torch.zeros(*residual.shape[:-1], 2, 2, dtype=residual.dtype, device=residual.device), active)
    _assert_tangent(residual, torch.ones_like(active) if dt is None else dt, active)
    dof = sum(2 * (int(row.sum()) - 1) for row in active.bool())
    if dof <= 0:
        raise ValueError("isotropic MLE requires at least two active intervals")
    return max(float((residual.square() * active.unsqueeze(-1)).sum() / dof), EPS_VAR)


def source_covariance(variant: str, calibrator: Calibrator, sigma_hist: torch.Tensor, n_eff: torch.Tensor, distance_ratio: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    if sigma_hist.ndim != 4 or sigma_hist.shape[-2:] != (2, 2) or sigma_hist.shape[:-2] != n_eff.shape:
        raise ValueError("sigma_hist must have shape [B,N,2,2]")
    if not torch.isfinite(sigma_hist).all():
        raise ValueError("sigma_hist must be finite")
    eye = torch.eye(2, dtype=sigma_hist.dtype, device=sigma_hist.device)
    s = evidence(n_eff, distance_ratio, available).unsqueeze(-1).unsqueeze(-1)
    if variant == "s0":
        return torch.zeros_like(sigma_hist)
    if variant == "s1":
        return calibrator.isotropic_variance * eye.expand_as(sigma_hist)
    if variant == "s2":
        return s * sigma_hist + ((1 - s) * calibrator.isotropic_variance + EPS_VAR) * eye
    if variant == "s3":
        return calibrator.alpha * s * sigma_hist + (calibrator.sigma_floor ** 2 + (1 - s) * calibrator.sigma_unknown ** 2) * eye
    raise ValueError("source variant must be s0/s1/s2/s3")


def _validate_calibration_shapes(residual: torch.Tensor, sigma_hist: torch.Tensor, active: torch.Tensor) -> None:
    if residual.ndim != 3 or residual.shape[-1] != 2 or sigma_hist.shape != (*residual.shape[:-1], 2, 2) or active.shape != residual.shape[:-1]:
        raise ValueError("invalid calibration tensor shapes")
    if not torch.isfinite(residual).all() or not torch.isfinite(sigma_hist).all():
        raise ValueError("calibration residual/covariance must be finite")


def _assert_tangent(residual: torch.Tensor, dt: torch.Tensor, active: torch.Tensor) -> None:
    if dt.shape != active.shape or torch.any(dt <= 0) or not torch.isfinite(dt).all() or not torch.all((active == 0) | (active == 1)):
        raise ValueError("invalid dt/active")
    error = (residual * dt.unsqueeze(-1) * active.unsqueeze(-1)).sum(dim=1).abs().max()
    if error >= 1e-6:
        raise ValueError("calibration residual violates endpoint tangent constraint")


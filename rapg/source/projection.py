"""Endpoint-constrained velocity projections."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def endpoint_residual(z: torch.Tensor, dt: torch.Tensor, delta_p: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    _validate_state(z, dt, delta_p, active)
    return (z * dt.unsqueeze(-1) * active.unsqueeze(-1)).sum(dim=1) - delta_p


def project_endpoint(
    z: torch.Tensor,
    dt: torch.Tensor,
    delta_p: torch.Tensor,
    active: torch.Tensor,
    w: torch.Tensor,
    *,
    return_jitter: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Project onto ``sum_i dt_i z_i = delta_p`` using free matrices ``W_i``.

    Direct Cholesky is attempted first; trace-relative jitter is added only on failure.
    """
    _validate_state(z, dt, delta_p, active)
    if w.shape != (*z.shape[:2], 2, 2):
        raise ValueError("w must have shape [B,N,2,2]")
    if not torch.isfinite(w).all():
        raise ValueError("w must be finite")
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in (z, dt, delta_p, w)) else torch.float32
    context = torch.autocast(device_type=z.device.type, enabled=False) if z.device.type in {"cpu", "cuda"} else nullcontext()
    with context:
        z, dt, delta_p, active, w = z.to(dtype), dt.to(dtype), delta_p.to(dtype), active.to(dtype), w.to(dtype)
        g = ((dt.square()).unsqueeze(-1).unsqueeze(-1) * w * active.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        residual = (z * dt.unsqueeze(-1) * active.unsqueeze(-1)).sum(dim=1) - delta_p
        inverse_residual, tries = solve_spd(g, residual)
        correction = -torch.einsum("bnij,bn,bj->bni", w, dt, inverse_residual) * active.unsqueeze(-1)
        out = z + correction
        _assert_closed(out, dt, delta_p, active)
    return (out, tries) if return_jitter else out


def tangent_project(field: torch.Tensor, dt: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Physical-time tangent projection; correction is constant on active slots."""
    if field.ndim != 3 or field.shape[-1] != 2 or dt.shape != field.shape[:2] or active.shape != dt.shape:
        raise ValueError("field/dt/active shapes must be [B,N,2], [B,N], [B,N]")
    _validate_dt_active(dt, active)
    if not torch.isfinite(field).all():
        raise ValueError("field must be finite")
    dtype = torch.float64 if field.dtype == torch.float64 or dt.dtype == torch.float64 else torch.float32
    context = torch.autocast(device_type=field.device.type, enabled=False) if field.device.type in {"cpu", "cuda"} else nullcontext()
    with context:
        x, d, a = field.to(dtype), dt.to(dtype), active.to(dtype)
        denominator = (d * a).sum(dim=1, keepdim=True)
        if torch.any(denominator <= 0):
            raise ValueError("active intervals require positive total dt")
        residual = (x * d.unsqueeze(-1) * a.unsqueeze(-1)).sum(dim=1, keepdim=True)
        out = x - residual / denominator.unsqueeze(-1) * a.unsqueeze(-1)
        if torch.max(torch.abs((out * d.unsqueeze(-1) * a.unsqueeze(-1)).sum(dim=1))) >= 1e-6:
            raise RuntimeError("tangent projection failed endpoint check")
    return out


def solve_spd(g: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve batched 2x2 SPD systems with relative trace jitter only on failure."""
    if g.ndim != 3 or g.shape[-2:] != (2, 2) or rhs.shape != (g.shape[0], 2):
        raise ValueError("g and rhs must have shapes [B,2,2] and [B,2]")
    if not torch.isfinite(g).all() or not torch.isfinite(rhs).all():
        raise ValueError("SPD system must be finite")
    g = (g + g.transpose(-1, -2)) * .5
    chol, info = torch.linalg.cholesky_ex(g)
    tries = torch.zeros(g.shape[0], dtype=torch.int64, device=g.device)
    failed = info.ne(0)
    if failed.any():
        base = g.clone()
        for retry, factor in enumerate((1e-7, 1e-5, 1e-3), start=1):
            indices = torch.nonzero(failed, as_tuple=False).flatten()
            if not len(indices):
                break
            trace_half = base[indices].diagonal(dim1=-2, dim2=-1).sum(dim=-1) * .5
            if torch.any(trace_half <= 0) or not torch.isfinite(trace_half).all():
                raise ValueError("SPD jitter requires positive finite trace")
            candidate = base[indices] + torch.eye(2, dtype=g.dtype, device=g.device).unsqueeze(0) * (trace_half * factor)[:, None, None]
            candidate_chol, candidate_info = torch.linalg.cholesky_ex(candidate)
            chol[indices] = candidate_chol
            tries[indices] = retry
            failed[indices] = candidate_info.ne(0)
        if failed.any():
            raise RuntimeError("rank-deficient/non-SPD system remains after relative jitter")
    return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1), tries


def _validate_state(z: torch.Tensor, dt: torch.Tensor, delta_p: torch.Tensor, active: torch.Tensor) -> None:
    if z.ndim != 3 or z.shape[-1] != 2 or dt.shape != z.shape[:2] or active.shape != dt.shape or delta_p.shape != (z.shape[0], 2):
        raise ValueError("state shapes must be z[B,N,2], dt/active[B,N], delta_p[B,2]")
    if not torch.isfinite(z).all() or not torch.isfinite(delta_p).all():
        raise ValueError("state must be finite")
    _validate_dt_active(dt, active)


def _validate_dt_active(dt: torch.Tensor, active: torch.Tensor) -> None:
    if not torch.isfinite(dt).all() or torch.any(dt <= 0):
        raise ValueError("dt must be finite and positive")
    if not torch.all((active == 0) | (active == 1)):
        raise ValueError("active must be binary")
    if torch.any(active.sum(dim=1) == 0):
        raise ValueError("every sample needs an active interval")


def _assert_closed(z: torch.Tensor, dt: torch.Tensor, delta_p: torch.Tensor, active: torch.Tensor) -> None:
    residual = (z * dt.unsqueeze(-1) * active.unsqueeze(-1)).sum(dim=1) - delta_p
    if torch.max(torch.abs(residual)) >= 1e-6:
        raise RuntimeError("endpoint projection failed closure check")

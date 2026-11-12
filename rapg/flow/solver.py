"""Constrained fixed-step Euler/Heun integration in float64 state space."""

import torch

from ..source import endpoint_residual, project_endpoint, tangent_project


def solve_flow(field, z0, dt, active, *, method="heun", steps=4, diagnostic=False):
    if steps < 1 or torch.any(dt <= 0):
        raise ValueError("positive dt and steps required")

    z = z0
    delta = (z0 * dt[:, :, None] * active[:, :, None]).sum(1)
    weights = torch.eye(2, dtype=z.dtype, device=z.device)[None, None] / dt[:, :, None, None]
    calls = []
    step_size = 1.0 / steps

    for k in range(steps):
        tau = torch.full((len(z),), k * step_size, dtype=z.dtype, device=z.device)
        f0 = tangent_project(field(z, tau), dt, active)
        calls.append(float(tau[0]))

        if method == "euler":
            z = project_endpoint(z + step_size * f0, dt, delta, active, weights)
        elif method == "heun":
            trial = project_endpoint(z + step_size * f0, dt, delta, active, weights)
            f1 = tangent_project(field(trial, tau + step_size), dt, active)
            calls.append(float((tau + step_size)[0]))
            z = project_endpoint(z + step_size * (f0 + f1) / 2.0, dt, delta, active, weights)
        else:
            raise ValueError(f"unsupported solver method: {method}")

        if not torch.isfinite(z).all() or endpoint_residual(z, dt, delta, active).abs().max() >= 1e-6:
            raise RuntimeError("ODE endpoint closure check failed")

    return (z, {"nfe": len(calls), "tau_calls": calls}) if diagnostic else z

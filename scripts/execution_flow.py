"""Training, ODE integration, and evaluation routines for TanFlow variants."""
from dataclasses import replace

import numpy as np
import torch

from rapg.eval.metrics import denormalize_xy, integrate_missing_positions
from rapg.research.rollout import bind_field, new_model
from rapg.source import endpoint_residual, project_endpoint, tangent_project
from scripts import multigap_data as d
from scripts import multigap_flow as base


ARMS = ("Main", "Off-Kin", "Bi-Noprior", "Bi-DR", "Final-Only",
        "Point-Source", "Unconstrained-Flow")
array_metrics = base.array_metrics


def source(c, data, rows, step, calibrator, arm="Main"):
    require_arm(arm)
    if arm == "Point-Source":
        return c.v0
    return base.source(c, data, rows, step, calibrator)


def require_arm(arm):
    if arm not in ARMS:
        raise ValueError("unknown experiment arm")


def require_geometry(active, gap):
    g = d.geometry(gap)
    if active.ndim != 2 or active.shape[1] != g["points"] - 1:
        raise ValueError("condition active mask shape differs from requested gap")
    expected = torch.zeros_like(active)
    expected[:, g["active_start"]:g["active_end_exclusive"]] = 1
    if not torch.equal(active, expected):
        raise ValueError("condition active mask differs from requested gap")


def conditions(data, gap, device, arm):
    require_arm(arm)
    require_geometry(data["active"], gap)
    c = d.training_conditions(data, gap, device)
    if arm == "Off-Kin":
        return replace(c, context=torch.zeros_like(c.context))
    if arm == "Bi-Noprior":
        duration = (c.dt * c.active).sum(1)
        chord = c.delta[:, None] / duration[:, None, None]
        return replace(c, v0=torch.where(c.active.bool()[:, :, None], chord, c.v0))
    return c


def mode(arm):
    require_arm(arm)
    return "direct_residual" if arm == "Bi-DR" else "flow"


def nfe(arm):
    return 1 if mode(arm) == "direct_residual" else 8


def objective(field, c, gt, z0, tau, arm):
    require_arm(arm)
    if arm == "Unconstrained-Flow":
        raise ValueError("Unconstrained-Flow is inference-only on Terminal-Only weights")
    if arm not in ("Bi-DR", "Final-Only"):
        return base.objective(field, c, gt, z0, tau, None, 0.)
    if not torch.isfinite(gt).all():
        raise ValueError("nonfinite velocity supervision")
    if arm == "Bi-DR":
        tau0 = torch.zeros(len(c.v0), dtype=c.v0.dtype, device=c.v0.device)
        output = tangent_project(field(c.v0, tau0), c.dt, c.active)
        target = gt - c.v0
    else:
        if z0 is None or tau is None or not torch.isfinite(z0).all():
            raise ValueError("finite closed FM source and tau required")
        teacher = (1 - tau[:, None, None]) * z0 + tau[:, None, None] * gt
        output = field(teacher, tau)
        target = gt - z0
    loss = (output - target).square()[c.active.bool()].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite active velocity loss")
    return loss, {"fm": float(loss.detach()), "rec": None, "field_calls": 1}


def final_only_heun(field, c):
    """Heun4 with active clamping, no stage projections, one final closure."""
    z, calls, stages = c.v0, [], []
    initial = c.v0
    mask = c.active[:, :, None]
    h = .25

    def call(state, tau):
        if not torch.isfinite(state).all():
            raise FloatingPointError("nonfinite Final-Only stage state")
        stages.append(endpoint_residual(state, c.dt, c.delta, c.active).detach())
        value = field(state, tau)
        if not torch.isfinite(value).all():
            raise FloatingPointError("nonfinite Final-Only field")
        calls.append(float(tau[0]))
        return value * mask

    for step in range(4):
        tau = torch.full((len(z),), step * h, dtype=z.dtype, device=z.device)
        f0 = call(z, tau)
        trial = z + h * f0
        f1 = call(trial, tau + h)
        z = z + h * (f0 + f1) / 2
        if not torch.isfinite(z).all():
            raise FloatingPointError("nonfinite Final-Only updated state")
        if not torch.equal(z[~c.active.bool()], initial[~c.active.bool()]):
            raise AssertionError("inactive intervals changed")
    pre_final = endpoint_residual(z, c.dt, c.delta, c.active).detach()
    w = torch.eye(2, dtype=z.dtype, device=z.device)[None, None] / c.dt[:, :, None, None]
    z = project_endpoint(z, c.dt, c.delta, c.active, w)
    return z, {"nfe": len(calls), "tau_calls": calls,
               "stage_residuals": torch.stack(stages, 1),
               "pre_final_residual": pre_final,
               "endpoint_checks": [float(x.abs().max()) for x in stages]}


def raw_heun(field, c):
    """Heun4 with active clamping and no tangent or terminal projection."""
    z, stages, calls = c.v0, [], []

    def call(state, tau):
        if not torch.isfinite(state).all():
            raise FloatingPointError("nonfinite raw-Heun state")
        stages.append(endpoint_residual(state, c.dt, c.delta, c.active).detach())
        calls.append(float(tau[0]))
        output = field(state, tau)
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite raw-Heun output")
        return output * c.active[:, :, None]

    for step in range(4):
        tau = torch.full((len(z),), step * .25, dtype=z.dtype, device=z.device)
        first = call(z, tau)
        trial = z + .25 * first
        second = call(trial, tau + .25)
        z = z + .125 * (first + second)
    final = endpoint_residual(z, c.dt, c.delta, c.active).detach()
    return z, {"nfe": len(calls), "tau_calls": calls,
               "stage_residuals": torch.stack(stages, 1), "pre_final_residual": final,
               "endpoint_checks": [float(x.abs().max()) for x in stages],
               "posthoc_closure": False, "stage_projection": False}


def predict(field, c, gap, arm="Main"):
    require_arm(arm)
    require_geometry(c.active, gap)
    if arm not in ("Bi-DR", "Final-Only", "Unconstrained-Flow"):
        return base.predict(field, c, gap)
    if arm == "Bi-DR":
        tau = torch.zeros(len(c.v0), dtype=c.v0.dtype, device=c.v0.device)
        raw = field(c.v0, tau)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite DR field")
        velocity = c.v0 + tangent_project(raw, c.dt, c.active)
        info = {"nfe": 1, "tau_calls": [0.], "endpoint_checks": []}
    elif arm == "Final-Only":
        velocity, info = final_only_heun(field, c)
    else:
        velocity, info = raw_heun(field, c)
    error = endpoint_residual(velocity, c.dt, c.delta, c.active).abs().max()
    if not torch.isfinite(velocity).all() or not torch.isfinite(error):
        raise FloatingPointError("nonfinite prediction or endpoint residual")
    if arm != "Unconstrained-Flow" and error >= 1e-6:
        raise FloatingPointError("final endpoint closure check failed")
    if not torch.equal(velocity[~c.active.bool()], c.v0[~c.active.bool()]):
        raise ValueError("inactive velocity intervals changed")
    info["endpoint_checks"].append(float(error))
    g = d.geometry(gap)
    positions = integrate_missing_positions(c.left, velocity, c.dt,
        start=g["anchor_left"], end=g["missing_end_inclusive"])
    if positions.shape[1:] != (gap, 2) or info["nfe"] != nfe(arm):
        raise ValueError("deployment geometry/NFE mismatch")
    return positions, velocity, info


@torch.no_grad()
def evaluate(model, data, gap, unit, device, batch_size, arm="Main"):
    require_arm(arm)
    if arm == "Main":
        return base.evaluate(model, data, gap, unit, device, batch_size)
    g = d.geometry(gap)
    was_training = model.training
    model.eval()
    predictions, truths, stages, pre_finals = [], [], [], []
    endpoint_norm, endpoint_raw = 0., 0.
    try:
        for start in range(0, len(data["sample_id"]), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(data["sample_id"])))
            batch = {key: value[rows] if isinstance(value, torch.Tensor)
                     and key not in ("coord_mean", "coord_std") else value
                     for key, value in data.items()}
            c = conditions(batch, gap, device, arm)
            pn, velocity, info = predict(bind_field(model, c), c, gap, arm)
            predictions.append(pn.cpu())
            truths.append(batch["trajectory"][:, :2,
                g["missing_start"]:g["anchor_right"]].transpose(1, 2).double())
            error = endpoint_residual(velocity, c.dt, c.delta, c.active)
            endpoint_norm = max(endpoint_norm, float(error.abs().max()))
            endpoint_raw = max(endpoint_raw, float(
                (error * data["coord_std"][:2].to(error)).abs().max()))
            if arm == "Final-Only":
                stages.append(info["stage_residuals"].cpu())
                pre_finals.append(info["pre_final_residual"].cpu())
    finally:
        model.train(was_training)
    pn, gn = torch.cat(predictions), torch.cat(truths)
    pred = denormalize_xy(pn, data["coord_mean"], data["coord_std"]).numpy()
    truth = denormalize_xy(gn, data["coord_mean"], data["coord_std"]).numpy()
    raw, norm = array_metrics(pred, truth), array_metrics(pn.numpy(), gn.numpy())
    metrics = {key: float(value.mean()) for key, value in raw.items()}
    metrics.update({f"norm_{key}": float(value.mean()) for key, value in norm.items()})
    metrics.update(nfe=nfe(arm), rows=len(pn), unit=unit, square_unit=f"{unit}^2",
        normalized_unit="B_normalized", gap=gap,
        endpoint_residual_normalized_inf=endpoint_norm, endpoint_residual_raw_inf=endpoint_raw)
    arrays = {"pred": pred, "truth": truth, "pred_norm": pn.numpy(), "truth_norm": gn.numpy(),
              "sample_id": data["sample_id"].numpy(), **raw,
              **{f"norm_{key}": value for key, value in norm.items()}}
    if "entity_id" in data:
        arrays["entity_id"] = data["entity_id"].numpy()
        arrays.update({f"lineage_{key}": value.numpy() for key, value in data["lineage"].items()})
    if stages:
        audit, pre = torch.cat(stages), torch.cat(pre_finals)
        metrics["stage_endpoint_drift_normalized_inf"] = float(audit.abs().max())
        metrics["pre_final_endpoint_drift_normalized_inf"] = float(pre.abs().max())
        arrays["stage_endpoint_residual_norm"] = audit.numpy()
        arrays["stage_endpoint_residual_raw"] = (audit * data["coord_std"][:2]).numpy()
        arrays["pre_final_endpoint_residual_norm"] = pre.numpy()
    return metrics, arrays

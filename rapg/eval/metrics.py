"""Position metrics in normalized and original XY coordinate spaces."""
from __future__ import annotations

import torch

METRICS_VERSION = "position_v2"
POSITION_KEYS = tuple(f"{space}_{metric}" for space in ("norm", "raw")
                      for metric in ("mse", "mae", "l2", "ade", "fde"))


def _positions(value: torch.Tensor, name: str) -> None:
    if value.ndim != 3 or value.shape[-1] != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must be nonempty [batch,points,2]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def denormalize_xy(positions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    _positions(positions, "positions")
    if mean.ndim != 1 or std.ndim != 1 or len(mean) < 2 or len(std) < 2:
        raise ValueError("coordinate statistics must contain x/y")
    mean, std = mean[:2].to(positions), std[:2].to(positions)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid coordinate normalization statistics")
    return positions * std + mean


def integrate_missing_positions(left_normalized, velocity_normalized, dt, start=9, end=14):
    _positions(velocity_normalized, "velocity")
    if (left_normalized.shape != (len(velocity_normalized), 2)
            or dt.shape != velocity_normalized.shape[:2]
            or not 0 <= start < end <= dt.shape[1]):
        raise ValueError("invalid position integration shapes or interval")
    if not torch.isfinite(left_normalized).all() or not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("left positions and positive dt must be finite")
    return left_normalized[:, None] + torch.cumsum(
        velocity_normalized[:, start:end] * dt[:, start:end, None], dim=1)


def position_metrics(pred, gt, prefix):
    """Per-window metrics; MSE is mean squared 2D distance, not per-coordinate MSE."""
    _positions(pred, "prediction"); _positions(gt, "ground truth")
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground truth shapes differ")
    diff = pred.double() - gt.double()
    distance = torch.linalg.vector_norm(diff, dim=-1)
    ade = distance.mean(1)
    return {f"{prefix}_mse": diff.square().sum(-1).mean(1),
            f"{prefix}_mae": diff.abs().mean((1, 2)),
            f"{prefix}_l2": ade, f"{prefix}_ade": ade,
            f"{prefix}_fde": distance[:, -1],
            f"{prefix}_max_point_error": distance.max(1).values,
            f"{prefix}_midpoint_error": distance[:, distance.shape[1] // 2]}


def metrics(pred, gt, pv, gv, active, endpoint, *, pred_normalized=None, gt_normalized=None):
    values = position_metrics(pred, gt, "raw")
    if (pred_normalized is None) != (gt_normalized is None):
        raise ValueError("both normalized coordinate tensors are required")
    if pred_normalized is not None:
        values.update(position_metrics(pred_normalized, gt_normalized, "norm"))
    result = {key: float(value.mean()) for key, value in values.items()}
    keep = active.bool(); speed = torch.linalg.vector_norm(pv, dim=-1); gs = torch.linalg.vector_norm(gv, dim=-1)
    valid = keep & (speed > 1e-6) & (gs > 1e-6)
    cos = (pv * gv).sum(-1) / (speed * gs).clamp_min(1e-12)
    result.update(metrics_version=METRICS_VERSION, ade=result["raw_ade"], fde=result["raw_fde"],
                  max_point_error=result["raw_max_point_error"], midpoint_error=result["raw_midpoint_error"],
                  velocity_mse=float((pv - gv).square()[keep].mean()),
                  speed_mae=float((speed - gs).abs()[keep].mean()),
                  direction_angular_error=float(torch.rad2deg(torch.acos(cos[valid].clamp(-1, 1))).mean()) if valid.any() else None,
                  direction_valid_count=int(valid.sum()), endpoint_residual_normalized_inf=float(endpoint.abs().max()))
    return result


def evaluate_baseline(data, kind):
    """Linear or cached kinematic prediction for the fixed mid5 protocol."""
    if kind not in ("linear", "kinematic"):
        raise ValueError("unknown baseline")
    dt, active, gt = data["dt"].double(), data["active"].double(), data["v_gt"].double()
    delta = data["delta_p"].double()
    if kind == "kinematic":
        velocity = data["v_closed"].double()
    else:
        duration = (dt * active).sum(1)
        if (duration <= 0).any():
            raise ValueError("Linear requires positive active duration")
        velocity = (delta / duration[:, None])[:, None] * active[..., None]
    pn = integrate_missing_positions(data["trajectory"][:, :2, 9].double(), velocity, dt)
    gn = data["trajectory"][:, :2, 10:15].transpose(1, 2).double()
    p = denormalize_xy(pn, data["coord_mean"], data["coord_std"])
    g = denormalize_xy(gn, data["coord_mean"], data["coord_std"])
    endpoint = (velocity * dt[..., None] * active[..., None]).sum(1) - delta
    result = metrics(p, g, velocity, gt, active, endpoint, pred_normalized=pn, gt_normalized=gn)
    per = {**position_metrics(pn, gn, "norm"), **position_metrics(p, g, "raw")}
    result.update(endpoint_residual_raw_inf=float((endpoint * data["coord_std"][:2]).abs().max()),
                  nfe_total=0, nfe_per_trajectory=0, source_projection_jitter_retries=0)
    result["per_sample"] = {"sample_id": data["sample_id"], "entity_id": data["entity_id"],
                            "pred_missing_xy_norm": pn, "gt_missing_xy_norm": gn,
                            "pred_missing_xy_raw": p, "gt_missing_xy_raw": g,
                            "ade": per["raw_ade"], **per}
    return result

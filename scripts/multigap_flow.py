"""Multi-gap tangent flow evaluation and trajectory metrics."""
import numpy as np
import torch

from rapg.eval.metrics import denormalize_xy, integrate_missing_positions
from rapg.flow import solve_flow
from rapg.research.rollout import Drivers, bind_field, new_model, objective
from rapg.source import endpoint_residual, source_covariance, tangent_project
from rapg.train.engine import _source
from scripts import multigap_data as d


def source(conditions, data, rows, step, calibrator):
    names = ("sigma_hist", "n_eff", "distance_ratio", "available")
    hist, ne, ratio, available = [data[key][rows].to(conditions.dt.device).double() for key in names]
    covariance = source_covariance("s2", calibrator, hist, ne, ratio, available)
    return _source(conditions.v0, conditions.dt, conditions.active, conditions.delta,
                   covariance, data["sample_id"][rows], step)


def predict(field, conditions, gap):
    g = d.geometry(gap)
    checks = []

    def checked_field(z, tau):
        error = endpoint_residual(z, conditions.dt, conditions.delta,
                                  conditions.active).abs().max()
        checks.append(float(error.detach()))
        if not torch.isfinite(z).all() or not torch.isfinite(error) or error >= 1e-6:
            raise FloatingPointError("intermediate endpoint closure check failed")
        output = field(z, tau)
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite field output")
        return tangent_project(output, conditions.dt, conditions.active)

    velocity, info = solve_flow(checked_field, conditions.v0, conditions.dt,
                                conditions.active, method="heun", steps=4, diagnostic=True)
    error = endpoint_residual(velocity, conditions.dt, conditions.delta,
                              conditions.active).abs().max()
    checks.append(float(error.detach()))
    if not torch.isfinite(velocity).all() or not torch.isfinite(error) or error >= 1e-6:
        raise FloatingPointError("prediction endpoint closure check failed")
    positions = integrate_missing_positions(conditions.left, velocity, conditions.dt,
                                            start=g["anchor_left"], end=g["missing_end_inclusive"])
    if positions.shape[1:] != (gap, 2) or info["nfe"] != 8:
        raise ValueError("deployment geometry/NFE mismatch")
    return positions, velocity, {**info, "endpoint_checks": checks}


def array_metrics(prediction, truth):
    pred, gt = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if (pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1] not in (5, 9, 13, 17, 21)
            or pred.shape[2] != 2 or len(pred) == 0):
        raise ValueError("invalid saved prediction shape")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("nonfinite saved prediction")
    diff = pred - gt
    error = np.linalg.norm(diff, axis=-1)
    return {"ade": error.mean(1), "fde": error[:, -1],
            "point_mse": (diff**2).sum(-1).mean(1),
            "coordinate_mae": np.abs(diff).mean((1, 2)),
            "midpoint": error[:, pred.shape[1] // 2], "maximum": error.max(1)}


@torch.no_grad()
def evaluate(model, data, gap, unit, device, batch_size):
    g = d.geometry(gap)
    was_training = model.training
    model.eval()
    raw_predictions, norm_predictions, norm_truths = [], [], []
    endpoint_norm, endpoint_raw = 0., 0.
    try:
        for start in range(0, len(data["sample_id"]), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(data["sample_id"])))
            batch = {key: value[rows] if isinstance(value, torch.Tensor)
                     and key not in ("coord_mean", "coord_std") else value
                     for key, value in data.items()}
            conditions = d.training_conditions(batch, gap, device)
            pn, velocity, info = predict(bind_field(model, conditions), conditions, gap)
            gn = batch["trajectory"][:, :2, g["missing_start"]:g["anchor_right"]].transpose(1, 2).double()
            raw_predictions.append(denormalize_xy(pn, data["coord_mean"], data["coord_std"]).cpu())
            norm_predictions.append(pn.cpu())
            norm_truths.append(gn.cpu())
            error = endpoint_residual(velocity, conditions.dt, conditions.delta, conditions.active)
            endpoint_norm = max(endpoint_norm, float(error.abs().max()))
            endpoint_raw = max(endpoint_raw, float((error * data["coord_std"][:2].to(error)).abs().max()))
            if info["nfe"] != 8:
                raise ValueError("deployment NFE differs from eight")
    finally:
        model.train(was_training)
    pn, gn = torch.cat(norm_predictions), torch.cat(norm_truths)
    pred = torch.cat(raw_predictions).numpy()
    truth = denormalize_xy(gn, data["coord_mean"], data["coord_std"]).numpy()
    raw_values, norm_values = array_metrics(pred, truth), array_metrics(pn.numpy(), gn.numpy())
    metrics = {key: float(values.mean()) for key, values in raw_values.items()}
    metrics.update({f"norm_{key}": float(values.mean()) for key, values in norm_values.items()})
    metrics.update(nfe=8, rows=len(data["sample_id"]), unit=unit, square_unit=f"{unit}^2",
                   normalized_unit="B_normalized", gap=gap,
                   endpoint_residual_normalized_inf=endpoint_norm,
                   endpoint_residual_raw_inf=endpoint_raw)
    arrays = {"pred": pred, "truth": truth, "pred_norm": pn.numpy(), "truth_norm": gn.numpy(),
              "sample_id": data["sample_id"].numpy(), **raw_values,
              **{f"norm_{key}": value for key, value in norm_values.items()}}
    if "entity_id" in data:
        arrays["entity_id"] = data["entity_id"].numpy()
        arrays.update({f"lineage_{key}": value.numpy() for key, value in data["lineage"].items()})
    return metrics, arrays

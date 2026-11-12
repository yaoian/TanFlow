"""Centered-gap window preparation and training-split source covariance fitting."""
import hashlib
import math

import torch

from rapg.data.prepare import _synthetic_sigma
from rapg.data.prior_velocity import build_kinematic_prior_velocity
from rapg.source import Calibrator, isotropic_mle, source_covariance
from rapg.source.calibration import EPS_VAR, evidence


DATASETS = ("atc", "edinburgh", "indoor", "sdd", "zara1", "zara2",
            "worldtrace_chengdu", "worldtrace_guangzhou", "worldtrace_shanghai")
SOURCE_KIND = "synthetic_train_global_uncentered_V_K_residual_second_moment"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(gap):
    if type(gap) is not int or gap not in (5, 9, 13, 17, 21):
        raise ValueError("only centered gaps5/9/13/17/21 are supported")
    start = (25 - gap) // 2
    return {"points": 25, "gap": gap, "missing_start": start,
            "missing_end_inclusive": start + gap - 1,
            "anchor_left": start - 1, "anchor_right": start + gap,
            "active_start": start - 1, "active_end_exclusive": start + gap,
            "mask_convention": "1=missing for prior; 1=observed in derived B contract"}


def missing_mask(gap, *, dtype=torch.float64, device=None):
    g = geometry(gap)
    mask = torch.zeros(25, dtype=dtype, device=device)
    mask[g["missing_start"]:g["anchor_right"]] = 1
    return mask


def targets(trajectory, gap):
    g = geometry(gap)
    values = trajectory.double()
    dt = values[:, 2, 1:] - values[:, 2, :-1]
    _require(values.ndim == 3 and values.shape[1:] == (3, 25)
             and torch.isfinite(values).all().item() and torch.all(dt > 0).item(),
             "truth must be finite [N,3,25] with strictly increasing time")
    active = torch.zeros_like(dt)
    active[:, g["active_start"]:g["active_end_exclusive"]] = 1
    velocity = ((values[:, :2, 1:] - values[:, :2, :-1]) / dt[:, None]).transpose(1, 2)
    gt = velocity * active[:, :, None]
    delta = values[:, :2, g["anchor_right"]] - values[:, :2, g["anchor_left"]]
    rho = torch.zeros_like(dt)
    q = dt[:, g["active_start"]:g["active_end_exclusive"]]
    rho[:, g["active_start"]:g["active_end_exclusive"]] = (
        q.cumsum(1) - q / 2) / q.sum(1, keepdim=True)
    return dt, gt, active, delta, rho


def obs_context(trajectory, dt, gap):
    g = geometry(gap)
    values = trajectory.double()
    durations = dt.to(device=values.device, dtype=torch.float64)
    _require(values.ndim == 3 and values.shape[1:] == (3, 25)
             and durations.shape == (len(values), 24), "invalid context shape")
    q = ~missing_mask(gap, device=values.device).bool()
    _require(torch.isfinite(values[:, :2, q]).all().item()
             and torch.isfinite(values[:, 2]).all().item()
             and torch.isfinite(durations).all().item() and (durations > 0).all().item(),
             "observed coordinates, times and positive durations must be finite")
    _require(torch.equal(values[:, 2, 1:] - values[:, 2, :-1], durations),
             "context durations differ from trajectory")
    # Select hidden XY out before differencing, including NaN and Inf probes.
    xy = torch.where(q[None, None], values[:, :2], torch.zeros_like(values[:, :2]))
    observed_intervals = (q[:-1] & q[1:]).double().expand(len(values), -1)
    velocity = (xy[:, :, 1:] - xy[:, :, :-1]) / durations[:, None]
    velocity = velocity * observed_intervals[:, None]
    _require(not observed_intervals[:, g["active_start"]:g["active_end_exclusive"]].any(),
             "hidden interval entered observation context")
    return torch.cat((velocity, observed_intervals[:, None]), 1)


def make_data(trajectory, identity, mean, std, gap):
    dt, gt, active, delta, rho = targets(trajectory, gap)
    prior = build_kinematic_prior_velocity(trajectory.double(), missing_mask(gap), mean, std)
    count = len(trajectory)
    result = {"sample_id": torch.arange(count), "trajectory": trajectory,
              "coord_mean": mean, "coord_std": std, "v_closed": prior, "v_gt": gt,
              "dt": dt, "active": active, "delta_p": delta, "rho": rho}
    if identity is not None:
        result["entity_id"] = torch.from_numpy(identity["trajectory_id"].copy())
        result["lineage"] = {key: torch.from_numpy(value.copy()) for key, value in identity.items()}
    return result


def source_fit(train, gap):
    residual, active = train["v_gt"] - train["v_closed"], train["active"]
    sigma, filler_iso = _synthetic_sigma(residual, active)
    variance = isotropic_mle(residual, active, train["dt"])
    dof = len(residual) * 2 * (gap + 1 - 1)
    raw = float((residual.square() * active[:, :, None]).sum() / dof)
    calibrator = Calibrator(1., math.sqrt(EPS_VAR), math.sqrt(variance), variance)
    n = len(train["sample_id"])
    ne = torch.full((1, 24), float(n), dtype=torch.float64)
    ratio, available = torch.ones_like(ne), torch.ones_like(ne)
    covariance = source_covariance("s2", calibrator, sigma[None], ne, ratio, available)
    keep = active[0].bool()
    eig = torch.linalg.eigvalsh(covariance[0, keep])
    return {"format": "s2_synthetic_train_global_fit_v1", "fit_split": "train",
            "source": "s2", "source_kind": SOURCE_KIND, "geometry": geometry(gap),
            "prior": "kinematic_V_K", "history": False, "retrieval": False,
            "train_rows": n, "uncentered": True, "second_moment_denominator": max(n - 1, 1),
            "fit_formula": "R_j.T @ R_j / max(N_train-1,1)",
            "isotropic_formula": "max(sum_active_xy((V_gt-V_K)^2)/(N_train*2*gap),1e-6)",
            "raw_isotropic_variance": raw, "degrees_of_freedom": dof,
            "floor_hit": raw <= EPS_VAR, "eps_var": EPS_VAR,
            "inactive_filler_variance": max(filler_iso, 1e-12),
            "sigma_hist": sigma.tolist(), "calibrator": calibrator.asdict(),
            "synthetic_evidence": {"n_eff": n, "distance_ratio": 1., "available": 1.,
                                   "weight": float(evidence(ne, ratio, available)[0, 0])},
            "active_source_covariance_eigenvalues": eig.tolist(),
            "active_source_eigenvalue_min": float(eig.min()),
            "active_source_eigenvalue_max": float(eig.max()),
            "active_source_max_anisotropy": float((eig[:, 1] / eig[:, 0]).max()),
            "source_exclusion_audit": None,
            "fit_population": "full supervised train split; no retrieval or OOF claim",
            "unused_s3_parameters": ["alpha", "sigma_floor", "sigma_unknown"]}


def attach_source(data, fit):
    count, n = len(data["sample_id"]), fit["train_rows"]
    return {**data,
            "n_eff": torch.full((1, 24), float(n), dtype=torch.float64).expand(count, 24),
            "distance_ratio": torch.ones(1, 24, dtype=torch.float64).expand(count, 24),
            "available": torch.ones(1, 24, dtype=torch.float64).expand(count, 24),
            "sigma_hist": torch.tensor(fit["sigma_hist"], dtype=torch.float64)[None].expand(count, 24, 2, 2)}


def training_conditions(data, gap, device):
    from rapg.research.rollout import Conditions
    g = geometry(gap)
    return Conditions(*(value.to(device=device, dtype=torch.float64) for value in
                        (data["v_closed"], data["dt"], data["active"],
                         obs_context(data["trajectory"], data["dt"], gap),
                         data["trajectory"][:, :2, g["anchor_left"]], data["delta_p"])))

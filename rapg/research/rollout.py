"""Eight-channel velocity field conditioning and flow-matching objective."""
from dataclasses import dataclass
import math, hashlib, json
import torch
from rapg.flow import FlowTCN, solve_flow
from rapg.eval.metrics import integrate_missing_positions
from rapg.source import endpoint_residual, tangent_project
from rapg.train.engine import _field_call

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_lambda(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("lambda must be finite and nonnegative")


def require_mode(checkpoint, requested):
    if requested not in ("flow", "direct_residual"):
        raise ValueError("unknown prediction mode")
    actual = checkpoint.get("prediction_mode", "flow")
    if actual != requested:
        raise ValueError(f"checkpoint prediction identity {actual} != {requested}")


def new_model(seed, mode="flow", device="cpu"):
    require_mode({"prediction_mode": mode}, mode)
    # Scratch initialization uses the historical order without advancing global RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FlowTCN(in_channels=8)
    model.obs_context = "bi"
    model.prior_mode = "kinematic"
    model.rho_mode = "off"
    model.z_channel = "state"
    model.prediction_mode = mode
    return model.to(device)


@dataclass
class Conditions:
    v0: torch.Tensor
    dt: torch.Tensor
    active: torch.Tensor
    context: torch.Tensor
    left: torch.Tensor
    delta: torch.Tensor

    def take(self, rows):
        return Conditions(*(getattr(self, k)[rows.to(self.dt.device)]
                            for k in self.__dataclass_fields__))


def bind_field(model, c):
    call = _field_call(model, "fm", c.context)
    rho = torch.zeros_like(c.dt)
    return lambda z, t: call(z, c.dt, rho, c.active, t)


def predict(field, c, mode="flow"):
    require_mode({"prediction_mode": mode}, mode)
    checks = []
    def checked_field(z, t):
        error = endpoint_residual(z, c.dt, c.delta, c.active).abs().max()
        checks.append(float(error.detach()))
        if not torch.isfinite(z).all() or not torch.isfinite(error) or error >= 1e-6:
            raise FloatingPointError("intermediate endpoint closure check failed")
        out = field(z, t)
        if not torch.isfinite(out).all():
            raise FloatingPointError("nonfinite field output")
        return out
    if mode == "flow":
        v, info = solve_flow(checked_field, c.v0, c.dt, c.active, method="heun", steps=4, diagnostic=True)
    else:
        tau = torch.zeros(len(c.v0), dtype=c.v0.dtype, device=c.v0.device)
        v = c.v0 + tangent_project(checked_field(c.v0, tau), c.dt, c.active)
        info = {"nfe": 1, "tau_calls": [0.]}
    error = endpoint_residual(v, c.dt, c.delta, c.active).abs().max()
    checks.append(float(error.detach()))
    if not torch.isfinite(v).all() or not torch.isfinite(error) or error >= 1e-6:
        raise FloatingPointError("prediction endpoint closure check failed")
    p = integrate_missing_positions(c.left, v, c.dt)
    return p, v, {**info, "endpoint_checks": checks}


def smooth_ade(pred, target):
    if target is None or pred.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError("finite separate supervision target required")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("nonfinite reconstruction")
    return ((pred - target).square().sum(-1).add(1e-8).sqrt() - 1e-4).mean()


def objective(field, c, gt_velocity, z0, tau, target_positions, weight):
    validate_lambda(weight)
    if not torch.isfinite(gt_velocity).all() or not torch.isfinite(z0).all():
        raise ValueError("nonfinite FM supervision/source")
    teacher = (1 - tau[:, None, None]) * z0 + tau[:, None, None] * gt_velocity
    output = tangent_project(field(teacher, tau), c.dt, c.active)
    fm = (output - (gt_velocity - z0)).square()[c.active.bool()].mean()
    if not torch.isfinite(fm):
        raise FloatingPointError("nonfinite FM loss")
    rec = None
    if weight:
        rec = smooth_ade(predict(field, c)[0], target_positions)
    return fm if rec is None else fm + weight * rec, {
        "fm": float(fm.detach()), "rec": None if rec is None else float(rec.detach()),
        "field_calls": 1 if rec is None else 9,
    }


class Drivers:
    """Private loader/tau streams and resumable, ordered driver hash chains."""
    def __init__(self, n, seed, batch_size=256, direct=False, calibration=False):
        offset = 200 if calibration else 0
        self.loader = torch.Generator().manual_seed(2026101100 + seed + offset)
        self.tau = torch.Generator().manual_seed(2026101200 + seed + offset)
        self.n, self.batch_size, self.direct = n, batch_size, direct
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.cursor = 0
        self.trace = {key: "0" * 64 for key in ("rows", "tau", "source")}
        self.last = None

    def next(self, step, sample_ids=None, source_step=None):
        if self.cursor >= len(self.permutation):
            self.permutation = torch.randperm(self.n, generator=self.loader)
            self.cursor = 0
        rows = self.permutation[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(rows)
        tau = None if self.direct else torch.rand(len(rows), generator=self.tau, dtype=torch.float64)
        ids = rows if sample_ids is None else sample_ids[rows]
        source_step = step if source_step is None else source_step
        seeds = [] if self.direct else [source_step * 1000003 + int(s) for s in ids]
        values = {"rows": rows.tolist(), "tau": [] if tau is None else tau.tolist(), "source": seeds}
        self.trace = {k: hashlib.sha256(bytes.fromhex(self.trace[k]) + canonical(values[k])).hexdigest()
                      for k in self.trace}
        self.last = {"step": step, "row_ids": rows.tolist(), "sample_ids": ids.tolist(),
                     "source_seeds": seeds, "tau_sha256": digest(values["tau"]), "rolling": dict(self.trace)}
        return rows, tau

    def state_dict(self):
        return {"loader": self.loader.get_state(), "tau": self.tau.get_state(),
                "permutation": self.permutation.clone(), "cursor": self.cursor,
                "trace": dict(self.trace), "n": self.n, "batch_size": self.batch_size,
                "direct": self.direct}

    def load_state_dict(self, state):
        if (state["n"], state["batch_size"], state["direct"]) != (self.n, self.batch_size, self.direct):
            raise ValueError("driver resume identity mismatch")
        self.loader.set_state(state["loader"])
        self.tau.set_state(state["tau"])
        self.permutation, self.cursor, self.trace = state["permutation"], state["cursor"], dict(state["trace"])


@dataclass
class Selection:
    best_score: float = float("inf")
    best_step: int = -1

    def update(self, step, score):
        if not math.isfinite(score):
            raise ValueError("nonfinite selection score")
        if score < self.best_score:
            self.best_score, self.best_step = score, step
            return True
        return False


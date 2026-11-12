"""Field binding and conditional source sampling primitives."""

from dataclasses import dataclass
import torch
from rapg.source import project_endpoint


@dataclass
class TrainConfig:
    prior_mode: str = "kinematic"
    obs_context: str = "bi"
    rho_mode: str = "off"
    model: str = "fm"
    source: str = "s2"

    def __post_init__(self):
        if self.prior_mode not in ("kinematic", "linear"):
            raise ValueError("Only retrieval-free prior modes are supported")


def _field(model, kind, z, dt, rho, a, tau=None, x_obs=None):
    if kind == "dr":
        return model(z.float(), dt.float(), rho.float(), a.float()).double()
    if x_obs is None:
        return model(z.float(), dt.float(), rho.float(), a.float(), tau.float()).double()
    return model(z.float(), dt.float(), rho.float(), a.float(), tau.float(), x_obs.float()).double()


def _field_call(model, kind, x_obs, z_channel=None):
    """Bind observation context and optional static state into the field callable."""
    bind = lambda z: z if z_channel is None else z_channel
    if x_obs is None:
        return lambda z, dt, rho, a, tau=None: _field(model, kind, bind(z), dt, rho, a, tau)
    return lambda z, dt, rho, a, tau=None: _field(model, kind, bind(z), dt, rho, a, tau, x_obs=x_obs)


def _source(vh, dt, a, d, cov, sid, seed, return_jitter=False):
    """Sample base trajectories conditioned on sample IDs and project to endpoints."""
    noise = torch.stack([
        torch.randn(vh.shape[1:], generator=torch.Generator().manual_seed(seed * 1000003 + int(x)), dtype=vh.dtype)
        for x in sid.detach().cpu()
    ]).to(vh.device)
    z = vh + torch.einsum("bnij,bnj->bni", torch.linalg.cholesky(cov), noise)
    return project_endpoint(z * a[:, :, None], dt, d, a, cov, return_jitter=return_jitter)

"""Non-causal TCN for direct residual and conditional flow fields."""

import math
import torch
from torch import nn
from ..source import tangent_project


class TimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freq", torch.exp(-math.log(10000) * torch.arange(32) / 31))
        self.net = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, tau):
        x = 1000 * tau[:, None] * self.freq[None]
        return self.net(torch.cat((x.sin(), x.cos()), 1))


class Block(nn.Module):
    def __init__(self, width, dilation, film):
        super().__init__()
        self.c1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.LayerNorm(width)
        self.n2 = nn.LayerNorm(width)
        self.film = nn.Linear(64, 2 * width) if film else None
        if self.film:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, time=None):
        h = self.n1(self.c1(x).transpose(1, 2)).transpose(1, 2)
        if self.film:
            scale, shift = self.film(time).chunk(2, 1)
            h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = torch.nn.functional.silu(h)
        h = self.n2(self.c2(h).transpose(1, 2)).transpose(1, 2)
        return x + torch.nn.functional.silu(h)


MODEL_SCALES = {"tiny": 48, "base": 64, "medium": 96, "large": 128, "xlarge": 192}


def scale_width(name: str) -> int:
    if name not in MODEL_SCALES:
        raise ValueError(f"unknown model_scale {name!r}; expected one of {sorted(MODEL_SCALES)}")
    return MODEL_SCALES[name]


class _TCN(nn.Module):
    def __init__(self, film=False, width=64, dilations=(1, 2, 4), in_channels=5, state_input=True):
        super().__init__()
        self.input = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.ModuleList([Block(width, d, film) for d in dilations])
        self.output = nn.Conv1d(width, 2, 1)
        self.state_input = state_input

    def run(self, z, dt, rho, active, time=None, x_obs=None):
        context = None if x_obs is None else x_obs.to(dt.dtype)
        if not self.state_input:
            x = torch.cat((dt[:, None], active[:, None].to(dt.dtype)), 1)
            return self._forward(x, time, context, active)
        z = z * active[:, :, None]
        rho = rho * active
        x = torch.cat((z.transpose(1, 2), dt[:, None], rho[:, None], active[:, None].to(z.dtype)), 1)
        return self._forward(x, time, context, active)

    def _forward(self, x, time, context, active):
        if context is not None:
            x = torch.cat((x, context), 1)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, time)
        return self.output(h).transpose(1, 2) * active[:, :, None]


class ResidualTCN(_TCN):
    """Direct Residual (DR) baseline network."""

    def __init__(self, width=64, dilations=(1, 2, 4)):
        super().__init__(False, width, dilations, 5)

    def forward(self, vh, dt, rho, active):
        return self.run(vh.float(), dt.float(), rho.float(), active.float())


class FlowTCN(_TCN):
    """Time-conditioned Flow Matching TCN backbone."""

    def __init__(self, width=64, dilations=(1, 2, 4), in_channels=8, state_input=True):
        super().__init__(True, width, dilations, in_channels, state_input)
        self.time = TimeMLP()

    def forward(self, z, dt, rho, active, tau, x_obs=None):
        return self.run(
            z.float(),
            dt.float(),
            rho.float(),
            active.float(),
            self.time(tau.float()),
            None if x_obs is None else x_obs.float(),
        )

"""Transformer backbone variant with matched parameter count for Table 2 ablation."""

import math

import torch
from torch import nn

from rapg.flow.model import TimeMLP

TARGET_PARAMETERS = 117122
ARCHITECTURES = {
    "Transformer": {
        "architecture": "Transformer_preLN_two_layers",
        "width": 64,
        "parameters": 117378,
        "nominal_receptive_field": 24,
        "global_attention": True,
        "heads": 4,
        "layers": 2,
        "feedforward": 256,
        "dropout": 0.0,
        "position_encoding": "fixed_sinusoidal_interval_index",
        "connectivity": "full self-attention over all 24 interval tokens, preLN residual attention and feedforward",
        "target_parameters": TARGET_PARAMETERS,
        "parameter_difference": 117378 - TARGET_PARAMETERS,
        "parameter_difference_percent": 100 * (117378 / TARGET_PARAMETERS - 1),
        "conditioning_channels": 8,
        "output_channels": 2,
    },
}


def field_input(z, dt, rho, active, x_obs):
    """Build the 8-channel model input tensor."""
    if x_obs is None or x_obs.shape != (len(z), 3, 24):
        raise ValueError("observed bilateral context x_obs [B, 3, 24] is required")
    if z.shape != (len(z), 24, 2) or dt.shape != active.shape or dt.shape != (len(z), 24):
        raise ValueError("expected 24 interval state/geometry tensors")
    z, dt, rho, active, context = (v.float() for v in (z, dt, rho, active, x_obs))
    return torch.cat(
        ((z * active[:, :, None]).transpose(1, 2), dt[:, None], (rho * active)[:, None], active[:, None], context),
        dim=1,
    )


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 64
        self.input = nn.Linear(8, width)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2)
        self.time = TimeMLP()
        position = torch.arange(24)[:, None]
        frequency = torch.exp(-math.log(10000.0) * torch.arange(0, width, 2) / width)
        encoding = torch.zeros(24, width)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", encoding[None])

    def forward(self, z, dt, rho, active, tau, x_obs=None):
        value = self.input(field_input(z, dt, rho, active, x_obs).transpose(1, 2))
        value = value + self.position + self.time(tau.float())[:, None]
        for layer in self.layers:
            value = layer(value)
        return self.output(self.norm(value)) * active.float()[:, :, None]


def build(name):
    if name != "Transformer":
        raise ValueError(f"unsupported backbone: {name}")
    model = Transformer()
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = ARCHITECTURES[name]["parameters"]
    if actual != expected:
        raise ValueError(f"unexpected parameter count for {name}: {actual} != {expected}")
    return model

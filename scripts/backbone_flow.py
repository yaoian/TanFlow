"""Backbone architecture ablation on the velocity flow-matching path."""

import torch

from scripts import backbone_models as models
from scripts import execution_flow

ARMS = ("Transformer",)
base = execution_flow.base
bind_field = execution_flow.bind_field
array_metrics = execution_flow.array_metrics


def architecture(arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return dict(models.ARCHITECTURES["Transformer"])


def conditions(data, gap, device, arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return execution_flow.conditions(data, gap, device, "Main")


def new_model(seed, *, arm="Transformer", device="cpu"):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = models.build("Transformer")
    model.obs_context = "bi"
    model.rho_mode = "off"
    model.z_channel = "state"
    model.prior_mode = "kinematic"
    model.prediction_mode = "flow"
    model.model_scale = "Transformer"
    return model.to(device)


def mode(arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return "flow"


def nfe(arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return 8


def source(c, data, rows, step, calibrator, arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return execution_flow.source(c, data, rows, step, calibrator)


def objective(field, c, gt, z0, tau, arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return execution_flow.objective(field, c, gt, z0, tau, "Main")


def predict(field, c, gap, arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return execution_flow.predict(field, c, gap, "Main")


def evaluate(model, data, gap, unit, device, batch_size, arm):
    if arm != "Transformer":
        raise ValueError(f"unknown backbone variant: {arm}")
    return execution_flow.evaluate(model, data, gap, unit, device, batch_size, "Main")

"""Unit and integration tests for TanFlow."""

import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rapg.eval.metrics import evaluate_baseline
from rapg.flow.model import FlowTCN, TimeMLP
from rapg.source import Calibrator, endpoint_residual, project_endpoint, tangent_project
from rapg.source.calibration import source_covariance
from scripts import multigap_data as data_helpers
from scripts.arms import ALIASES, ARMS, Arm
from scripts.release_run import demo_data, frechet_distances, prepare
from scripts.release_run import main as release_main

torch.set_num_threads(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_architectures_and_parameter_counts():
    """Verify default constructors and exact parameter counts across all model scales."""
    expected = {
        "FlowTCN-default": (FlowTCN(), 117122),
        "TanFlow-Full": (FlowTCN(width=64, in_channels=8), 117122),
        "Direct-TCN": (Arm.resolve("Direct-TCN").model(seed=0, device="cpu"), 117122),
    }
    for name, (model, target_params) in expected.items():
        actual = count_params(model)
        assert actual == target_params, f"{name}: expected {target_params}, got {actual}"
    assert inspect.signature(FlowTCN).parameters["in_channels"].default == 8
    assert FlowTCN().input.in_channels == 8


def test_affine_geometry_closure_and_projections():
    """Verify affine endpoint closure and tangent projection residual < 1e-12."""
    torch.manual_seed(0)
    b, n = 4, 24
    dt = torch.rand(b, n, dtype=torch.float64) * 0.5 + 0.2
    active = torch.zeros(b, n, dtype=torch.float64)
    active[:, 5:19] = 1.0  # mid13: M=13 missing points -> M+1=14 active intervals
    delta_p = torch.randn(b, 2, dtype=torch.float64)
    v_raw = torch.randn(b, n, 2, dtype=torch.float64)
    w = (1.0 / dt).unsqueeze(-1).unsqueeze(-1) * torch.eye(2, dtype=torch.float64).view(1, 1, 2, 2)

    v_closed = project_endpoint(v_raw, dt, delta_p, active, w)
    res = endpoint_residual(v_closed, dt, delta_p, active)
    assert float(res.abs().max()) < 1e-12, f"Endpoint closure residual too large: {res.abs().max()}"

    u_tan = tangent_project(v_raw, dt, active)
    zero_delta = torch.zeros_like(delta_p)
    res_tan = endpoint_residual(u_tan, dt, zero_delta, active)
    assert float(res_tan.abs().max()) < 1e-12, f"Tangent projection residual too large: {res_tan.abs().max()}"


def test_arms_resolution():
    """Verify the exact Table 2 registry, model construction, and aliases."""
    assert ARMS == (
        "TanFlow-Full", "Terminal-Only", "Unconstrained-Flow", "Chord-Source",
        "Point-Source", "Direct-TCN", "Transformer",
    )
    for name in ARMS:
        arm = Arm.resolve(name)
        assert arm.name == name
        assert arm.model(seed=0, device="cpu") is not None, (
            f"Variant failed to instantiate: {name}")
    expected_aliases = {
        "TanFlow": "TanFlow-Full",
        "TanFlow-Base": "TanFlow-Full",
        "Main": "TanFlow-Full",
        "Final-Only": "Terminal-Only",
        "No-Proj": "Unconstrained-Flow",
        "NoProjection": "Unconstrained-Flow",
        "Bi-Linear": "Chord-Source",
        "Bi-Noprior": "Chord-Source",
        "S0": "Point-Source",
        "Bi-DR": "Direct-TCN",
        "1-Step-Direct-TCN": "Direct-TCN",
    }
    assert ALIASES == expected_aliases
    for alias, canonical in ALIASES.items():
        assert Arm.resolve(alias).name == canonical


def test_protocol_constants():
    """Check scientific constants directly against model architecture."""
    from rapg.research.rollout import Conditions, bind_field, new_model

    model = new_model(0)
    assert model.rho_mode == "off" and model.prior_mode == "kinematic"
    assert model.input.in_channels == inspect.signature(FlowTCN).parameters["in_channels"].default == 8
    b, n = 2, 24
    z = torch.randn(b, n, 2)
    dt = torch.ones(b, n)
    active = torch.ones(b, n)
    context = torch.randn(b, 3, n)
    conditions = Conditions(z, dt, active, context, torch.zeros(b, 2), z.sum(1))
    captured = []
    hook = model.input.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    bind_field(model, conditions)(z, torch.zeros(b))
    hook.remove()
    assert captured and captured[0].shape == (b, 8, n)
    assert not captured[0][:, 3].any(), "reserved rho_sup channel must remain identically zero"
    torch.testing.assert_close(captured[0][:, 5:], context)

    time_mlp = model.time
    assert isinstance(time_mlp, TimeMLP) and time_mlp.freq.numel() == 32
    assert (time_mlp.net[0].in_features, time_mlp.net[0].out_features) == (64, 128)


def test_default_forward_and_reserved_channel():
    model = FlowTCN()
    b, n = 2, 24
    z = torch.randn(b, n, 2)
    dt = torch.ones(b, n)
    active = torch.ones(b, n)
    context = torch.randn(b, 3, n)
    captured = []
    hook = model.input.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    from rapg.research.rollout import Conditions, bind_field
    conditions = Conditions(z, dt, active, context, torch.zeros(b, 2), z.sum(1))
    output = bind_field(model, conditions)(z, torch.zeros(b))
    hook.remove()
    assert output.shape == (b, n, 2)
    assert captured[0].shape == (b, 8, n)
    assert not captured[0][:, 3].any()
    torch.testing.assert_close(captured[0][:, 5:], context)


@pytest.mark.parametrize("count", [3, 8, 20])
def test_uncentered_s2_denominator_dof_and_evidence(count):
    gap = 13
    dt = torch.linspace(.2, .9, 24, dtype=torch.float64).expand(count, -1)
    active = torch.zeros_like(dt)
    active[:, 5:19] = 1
    residual = torch.zeros(count, 24, 2, dtype=torch.float64)
    residual[:, 5, 0] = torch.arange(1, count + 1, dtype=torch.float64)
    residual[:, 18, 0] = -residual[:, 5, 0] * dt[:, 5] / dt[:, 18]
    train = {"v_gt": residual, "v_closed": torch.zeros_like(residual),
             "active": active, "dt": dt, "sample_id": torch.arange(count)}
    fit = data_helpers.source_fit(train, gap)
    second_moment = residual[:, 5].T @ residual[:, 5] / (count - 1)
    torch.testing.assert_close(torch.tensor(fit["sigma_hist"][5]), second_moment.float())
    centered = residual[:, 5] - residual[:, 5].mean(0)
    assert not torch.allclose(second_moment, centered.T @ centered / (count - 1))
    expected_iso = max(float(residual.square().sum()) / (count * 2 * gap), 1e-6)
    assert fit["degrees_of_freedom"] == count * 2 * gap
    assert fit["calibrator"]["isotropic_variance"] == pytest.approx(expected_iso)
    weight = 0 if count < 4 else min(count / 16, 1) * math.exp(-.5)
    assert fit["synthetic_evidence"]["weight"] == pytest.approx(weight)
    cov = source_covariance(
        "s2", Calibrator(**fit["calibrator"]),
        torch.tensor(fit["sigma_hist"], dtype=torch.float64)[None],
        torch.full((1, 24), float(count), dtype=torch.float64),
        torch.ones(1, 24, dtype=torch.float64), torch.ones(1, 24, dtype=torch.float64))
    expected = weight * second_moment + ((1 - weight) * expected_iso + 1e-6) * torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(cov[0, 5], expected)


def test_prior_context_do_not_read_hidden_coordinates(tmp_path):
    path = tmp_path / "data.npz"
    demo_data(path)
    train, _, _, _, _, _ = prepare(path, "atc", "TanFlow-Full", 13)
    original = train[13]
    corrupt = original["trajectory"].clone()
    corrupt[:, :2, 6:19] = float("nan")
    from rapg.data.prior_velocity import build_kinematic_prior_velocity
    prior = build_kinematic_prior_velocity(
        corrupt, data_helpers.missing_mask(13), original["coord_mean"], original["coord_std"])
    torch.testing.assert_close(prior, original["v_closed"])
    torch.testing.assert_close(data_helpers.obs_context(corrupt, original["dt"], 13),
                               data_helpers.obs_context(original["trajectory"], original["dt"], 13))


def test_chord_source_center(tmp_path):
    path = tmp_path / "data.npz"
    demo_data(path)
    full, _, _, _, _, _ = prepare(path, "atc", "TanFlow-Full", 13)
    data = full[13]
    conditions = Arm.resolve("Chord-Source").conditions(data, 13, "cpu")
    expected = data["delta_p"][:, None] / (data["dt"] * data["active"]).sum(1)[:, None, None]
    expected = expected * data["active"][:, :, None]
    torch.testing.assert_close(conditions.v0, expected)
    assert conditions.context.any()


def test_projection_intervention_causal_scope(tmp_path):
    path = tmp_path / "data.npz"
    demo_data(path)
    train, _, _, _, _, _ = prepare(path, "atc", "TanFlow-Full", 13)
    full_arm = Arm.resolve("TanFlow-Full")
    terminal_arm = Arm.resolve("Terminal-Only")
    unconstrained_arm = Arm.resolve("Unconstrained-Flow")
    c = full_arm.conditions(train[13], 13, "cpu")
    field = lambda z, tau: torch.ones_like(z) * c.active[:, :, None]
    tau = torch.full((len(c.v0),), .3, dtype=torch.float64)
    full_loss, _ = full_arm.objective(field, c, c.v0, c.v0, tau)
    raw_loss, _ = terminal_arm.objective(field, c, c.v0, c.v0, tau)
    assert full_loss < 1e-25
    assert raw_loss == 1
    raw, raw_info = unconstrained_arm.module.raw_heun(field, c)
    closed, closed_info = terminal_arm.module.final_only_heun(field, c)
    assert raw_info["nfe"] == closed_info["nfe"] == 8
    torch.testing.assert_close(raw_info["stage_residuals"], closed_info["stage_residuals"])
    weights = torch.eye(2)[None, None] / c.dt[:, :, None, None]
    torch.testing.assert_close(closed, project_endpoint(raw, c.dt, c.delta, c.active, weights))
    assert endpoint_residual(raw, c.dt, c.delta, c.active).abs().max() > 1
    assert endpoint_residual(closed, c.dt, c.delta, c.active).abs().max() < 1e-12
    with pytest.raises(ValueError, match="inference-only"):
        unconstrained_arm.objective(field, c, c.v0, c.v0, tau)


def test_schur_rank_and_prefix_affine_inverse():
    generator = torch.Generator().manual_seed(17)
    m = 13
    h = torch.rand(m + 1, generator=generator, dtype=torch.float64) + .2
    a = torch.cat([value * torch.eye(2, dtype=torch.float64) for value in h], 1)
    blocks = torch.randn(m + 1, 2, 2, generator=generator, dtype=torch.float64)
    covariance = torch.block_diag(*(b @ b.T + .1 * torch.eye(2) for b in blocks))
    q = torch.eye(2 * (m + 1)) - covariance @ a.T @ torch.linalg.solve(a @ covariance @ a.T, a)
    tangent_cov = q @ covariance @ q.T
    assert torch.linalg.matrix_rank(tangent_cov) == 2 * m
    assert (a @ tangent_cov).abs().max() < 1e-12
    left, right = torch.tensor([1., 2.]), torch.tensor([4., -1.])
    positions = torch.randn(m, 2, generator=generator, dtype=torch.float64)
    complete = torch.cat((left[None], positions, right[None]))
    velocity = complete.diff(dim=0) / h[:, None]
    restored = left + (velocity * h[:, None]).cumsum(0)[:-1]
    torch.testing.assert_close(restored, positions)
    torch.testing.assert_close(a @ velocity.flatten(), (right - left).double())


def test_kinematic_baseline_name_and_legacy_rejection(tmp_path):
    path = tmp_path / "data.npz"
    demo_data(path)
    train, _, _, _, _, _ = prepare(path, "atc", "TanFlow-Full", 5)
    data = train[5]
    data["entity_id"] = torch.arange(len(data["sample_id"]))
    assert math.isfinite(evaluate_baseline(data, "kinematic")["ade"])
    with pytest.raises(ValueError, match="unknown baseline"):
        evaluate_baseline(data, "".join(("ha", "kp")))


def test_end_to_end_smoke_train_and_evaluate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "synthetic_atc.npz"
        run_dir = root / "run_full"
        eval_dir = root / "eval_full"

        release_main(["demo-data", "--output", str(data_path)])
        release_main([
            "train",
            "--data", str(data_path),
            "--output", str(run_dir),
            "--dataset", "atc",
            "--arm", "TanFlow-Full",
            "--gap", "13",
            "--seed", "0",
            "--steps", "4",
            "--eval-every", "2",
            "--batch-size", "8",
            "--smoke",
        ])
        assert (run_dir / "best.pt").is_file()
        assert (run_dir / "complete.json").is_file()

        release_main([
            "evaluate",
            "--data", str(data_path),
            "--checkpoint", str(run_dir / "best.pt"),
            "--output", str(eval_dir),
            "--allow-smoke",
        ])
        assert (eval_dir / "metrics.json").is_file()
        assert (eval_dir / "mid13_predictions.npz").is_file()


def test_terminal_only_checkpoint_reused_for_unconstrained_flow(tmp_path):
    path, run = tmp_path / "data.npz", tmp_path / "terminal_only"
    demo_data(path)
    release_main(["train", "--data", str(path), "--output", str(run),
                  "--dataset", "atc", "--arm", "Terminal-Only", "--gap", "13",
                  "--steps", "2", "--eval-every", "1", "--batch-size", "8", "--smoke"])
    frozen_hash = sha256(run / "best.pt")
    for arm in ("Terminal-Only", "Unconstrained-Flow"):
        output = tmp_path / arm
        release_main(["evaluate", "--data", str(path), "--checkpoint", str(run / "best.pt"),
                      "--output", str(output), "--arm", arm, "--allow-smoke"])
        freeze = json.loads((output / "selection_frozen_before_test.json").read_text())
        assert freeze["checkpoint_sha256"] == frozen_hash
        assert freeze["selection_arm"] == "Terminal-Only"
        assert not freeze["independently_selected_unconstrained_flow"]
    assert sha256(run / "best.pt") == frozen_hash


def test_discrete_frechet_matches_reference():
    rng = np.random.default_rng(3)
    query = rng.normal(size=(4, 2))
    candidates = rng.normal(size=(3, 4, 2))
    expected = []
    for candidate in candidates:
        coupling = np.full((4, 4), np.inf)
        for i in range(4):
            for j in range(4):
                distance = np.linalg.norm(query[i] - candidate[j])
                if i == j == 0:
                    coupling[i, j] = distance
                elif i == 0:
                    coupling[i, j] = max(coupling[i, j - 1], distance)
                elif j == 0:
                    coupling[i, j] = max(coupling[i - 1, j], distance)
                else:
                    coupling[i, j] = max(min(coupling[i - 1, j], coupling[i - 1, j - 1], coupling[i, j - 1]), distance)
        expected.append(coupling[-1, -1])
    np.testing.assert_allclose(frechet_distances(query, candidates), expected)


def test_query_cli_contract_and_evaluation(tmp_path):
    tiled = np.repeat(np.repeat(np.linspace(0, 24, 25)[:, None], 2, axis=1)[None], 12, axis=0)
    np.save(tmp_path / "positions.npy", tiled)
    np.save(tmp_path / "times.npy", np.repeat(np.arange(25, dtype=float)[None], 12, axis=0))
    np.savez(tmp_path / "predictions.npz", pred=np.zeros((12, 13, 2)), truth=np.zeros((12, 13, 2)))
    workload = dict(spatial_lower=[[-1., -1.]], spatial_upper=[[25., 1.]],
                    time_start=[0.], time_end=[24.], knn_query_indices=[0],
                    stay_speed_threshold=2., stay_min_duration=1.,
                    normalization_mean_xy=[0., 0.], normalization_scale_xy=[1., 1.])
    (tmp_path / "workload.json").write_text(json.dumps(workload))
    release_main(["evaluate-queries", "--predictions", str(tmp_path / "predictions.npz"),
                  "--positions", str(tmp_path / "positions.npy"), "--times", str(tmp_path / "times.npy"),
                  "--workload", str(tmp_path / "workload.json"), "--output", str(tmp_path / "result")])
    result = json.loads((tmp_path / "result/query_metrics.json").read_text())
    assert result["range_F1"] == result["knn_Recall_at_10"] == 1
    assert result["stay_duration_MAE"] == 0

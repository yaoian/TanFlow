"""Training and evaluation CLI for TanFlow.

Provides subcommands for synthetic smoke data generation (``demo-data``), model
training (``train``), trajectory gap evaluation (``evaluate``), and downstream
spatial-temporal query evaluation (``evaluate-queries``).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from rapg.eval.metrics import integrate_missing_positions
from rapg.research.rollout import Drivers
from rapg.source import Calibrator, endpoint_residual
from scripts import multigap_data as d
from scripts.arms import ARMS, Arm, gaps_for


def configure():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*(root / "rapg").rglob("*.py"), *(root / "scripts").rglob("*.py")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_data(path, splits):
    """NPZ is lazy: training requests only train/val, never the test array."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"].item()))
        if meta.get("coordinates") != "B_normalized" or meta.get("normalization_fit") != "full_train":
            raise ValueError("Data must use full-train B_normalized coordinates")
        mean = torch.tensor([*meta["mean_xy"], 0.], dtype=torch.float64)
        std = torch.tensor([*meta["xy_scale"], meta["time_scale"]], dtype=torch.float64)
        if (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all() or std[0] != std[1]):
            raise ValueError("Normalization must be finite, positive and spatially isotropic")
        arrays, keys = {}, {}
        for split in splits:
            values = f[split]
            if (values.dtype != np.float32 or values.ndim != 3 or values.shape[1:] != (25, 3)
                    or not len(values) or not np.isfinite(values).all()
                    or not (np.diff(values[:, :, 2], axis=1) > 0).all()):
                raise ValueError(f"Invalid {split}: expected finite float32 [N,25,3]")
            arrays[split] = torch.from_numpy(values.copy()).transpose(1, 2).contiguous()
            name = split + "_entity"
            keys[split] = f[name].astype(str).tolist() if name in f.files else None
            if keys[split] is not None and len(keys[split]) != len(values):
                raise ValueError("Entity keys must follow the exact window order")
        if meta.get("entity_disjoint") is True:
            for i, a in enumerate(splits):
                for b in splits[i + 1:]:
                    if keys[a] is None or keys[b] is None or set(keys[a]) & set(keys[b]):
                        raise ValueError("Claimed entity-disjoint input has missing/overlapping identities")
    return arrays, keys, mean, std, meta


@torch.no_grad()
def kin_train_scale(data, gap):
    """Train-only kinematic-prior error used as the selection denominator."""
    g = d.geometry(gap)
    positions = integrate_missing_positions(
        data["trajectory"][:, :2, g["anchor_left"]].double(),
        data["v_closed"], data["dt"], start=g["anchor_left"],
        end=g["missing_end_inclusive"])
    truth = data["trajectory"][:, :2, g["missing_start"]:g["anchor_right"]].transpose(1, 2).double()
    error = ((positions - truth) * data["coord_std"][:2]).numpy()
    value = float(np.sqrt(np.square(error).sum(-1)).mean())
    if not np.isfinite(value):
        raise ValueError("invalid train-only Kin selection denominator")
    return max(value, 1e-12)


def prepare(path, dataset, arm, gap):
    arrays, _, mean, std, meta = load_data(path, ("train", "val"))
    if meta["dataset"] != dataset:
        raise ValueError("Dataset metadata does not match --dataset")
    selection = {"sampling_unit": "full_train", "selected_windows": len(arrays["train"])}
    rows = np.arange(len(arrays["train"]), dtype=np.int64)
    train, val, fits, denominators = {}, {}, {}, {}
    for current_gap in gaps_for(arm, gap):
        tr = d.make_data(arrays["train"][rows], None, mean, std, current_gap)
        tr["source_sample_id"] = torch.from_numpy(rows.copy())
        fit = d.source_fit(tr, current_gap)
        fit["fit_population"] = "selected training windows only; frozen full-train normalization"
        fit["source_original_sample_ids"] = rows.tolist()
        tr = d.attach_source(tr, fit)
        va = d.attach_source(d.make_data(arrays["val"], None, mean, std, current_gap), fit)
        train[current_gap], val[current_gap], fits[current_gap] = tr, va, fit
        denominators[current_gap] = kin_train_scale(tr, current_gap)
    return train, val, fits, denominators, selection, meta


@torch.no_grad()
def evaluate(model, arm, data, gap, device="cpu", batch=256):
    if model is not None:
        model.eval()
    predictions, residuals, stages = [], [], []
    g = d.geometry(gap)
    for start in range(0, len(data["sample_id"]), batch):
        rows = torch.arange(start, min(start + batch, len(data["sample_id"])))
        subset = {key: (value[rows] if isinstance(value, torch.Tensor)
                       and key not in ("coord_mean", "coord_std") else value)
                  for key, value in data.items()}
        conditions = arm.conditions(subset, gap, device)
        pred, velocity, info = arm.predict(model, conditions, gap)
        active = getattr(conditions, "velocity_active", conditions.active)
        predictions.append(pred.cpu())
        residuals.append(endpoint_residual(velocity, conditions.dt, conditions.delta, active).cpu())
        if "stage_residuals" in info:
            stages.append(info["stage_residuals"].cpu())
    prediction = torch.cat(predictions).numpy()
    truth = data["trajectory"][:, :2, g["missing_start"]:g["anchor_right"]].transpose(1, 2).double().numpy()
    mean, scale = data["coord_mean"][:2].numpy(), data["coord_std"][:2].numpy()
    native_pred, native_truth = prediction * scale + mean, truth * scale + mean
    metrics = d_metrics(native_pred, native_truth)
    metrics.update(nfe=info["nfe"], rows=len(truth),
                   endpoint_residual_native_inf=float(np.abs(torch.cat(residuals).numpy()*scale).max()))
    result = dict(pred=native_pred, truth=native_truth, pred_norm=prediction, truth_norm=truth,
                  sample_id=data["sample_id"].numpy(),
                  endpoint_residual_normalized=torch.cat(residuals).numpy())
    if stages:
        result["stage_residuals_normalized"] = torch.cat(stages).numpy()
    return metrics, result


def d_metrics(pred, truth):
    delta = pred - truth
    error = np.linalg.norm(delta, axis=-1)
    return dict(ade=float(error.mean()), fde=float(error[:, -1].mean()),
                point_mse=float(np.square(delta).sum(-1).mean()),
                coordinate_mse=float(np.square(delta).mean()),
                coordinate_mae=float(np.abs(delta).mean()),
                midpoint=float(error[:, error.shape[1]//2].mean()),
                maximum=float(error.max(1).mean()))


def train(args):
    configure()
    arm = Arm.resolve(args.arm)
    if arm.name == "Unconstrained-Flow":
        raise ValueError(
            "Unconstrained-Flow is evaluation-only; use a Terminal-Only checkpoint")
    budget = args.steps if args.smoke else (60000 if args.dataset == "sdd" else 30000)
    if args.steps is not None and not args.smoke:
        raise ValueError("Custom update counts require --smoke")
    if budget is None or budget < 1 or args.eval_every < 1 or args.batch_size < 1:
        raise ValueError("Positive steps, evaluation period and batch size required")
    if not args.smoke and (args.eval_every != 500 or args.batch_size != 256):
        raise ValueError("Formal runs require batch256 and validation every500")
    learning_rate = args.lr if args.lr is not None else 3e-4
    if not args.smoke and learning_rate != 3e-4:
        raise ValueError("Formal paper variants require learning rate 3e-4")
    directory = Path(args.output)
    if directory.exists() and not args.resume:
        raise FileExistsError("Output exists; use explicit --resume with unchanged inputs/config")
    if args.resume and not (directory / "last.pt").is_file():
        raise FileNotFoundError("Resume requires last.pt")
    config = dict(arm=arm.name, dataset=args.dataset, gap=args.gap, seed=args.seed, steps=budget,
                  batch_size=args.batch_size, eval_every=args.eval_every, lr=learning_rate,
                  smoke=args.smoke, data_sha256=sha(args.data), code_sha256=code_hash(),
                  device=args.device, prior_mode="kinematic", context="bi", rho="off")
    tr, va, fits, denominators, selection, meta = prepare(args.data, args.dataset, arm.name, args.gap)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = arm.model(args.seed, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    drivers = {gap: Drivers(len(data["sample_id"]), args.seed, args.batch_size,
                           direct=arm.name == "Direct-TCN") for gap, data in tr.items()}
    conditions = {gap: arm.conditions(data, gap, args.device) for gap, data in tr.items()}
    calibrators = {gap: Calibrator(**fit["calibrator"]) for gap, fit in fits.items()}
    curve, best, begin = [], None, 1
    if args.resume:
        state = torch.load(directory / "last.pt", weights_only=False, map_location="cpu")
        if state["config"] != config:
            raise ValueError("Checkpoint config, code, or data hash mismatch on resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for gap, driver in drivers.items():
            driver.load_state_dict(state["drivers"][gap])
        torch.set_rng_state(state["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        begin, best, curve = state["step"] + 1, state["best"], state["curve"]
    else:
        save_json(directory / "config.json", config)
        save_json(directory / "population.json", selection)
        save_json(directory / "source_fits.json", fits)
        save_json(directory / "environment.json", dict(
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            device=args.device, orchestration="TanFlow evaluation and reproduction runner"))

    def checkpoint(step):
        return dict(config=config, model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    step=step, best=best, curve=curve, normalization=meta,
                    source_fits=fits, denominators=denominators)

    def validate(step):
        nonlocal best
        scores = {gap: evaluate(model, arm, data, gap, args.device)[0] for gap, data in va.items()}
        score = next(iter(scores.values()))["ade"]
        if not math.isfinite(score):
            raise FloatingPointError("Validation returned a nonfinite selection score")
        curve.append(dict(step=step, score=score, by_gap=scores))
        if step > 0 and (best is None or score < best["score"]):
            best = dict(step=step, score=score)
            atomic_save(directory / "best.pt", checkpoint(step))
        save_json(directory / "curve.json", curve)

    if begin == 1:
        validate(0)
    for step in range(begin, budget + 1):
        gap = sorted(tr)[(step - 1) % len(tr)]
        data, driver = tr[gap], drivers[gap]
        rows, tau = driver.next(step, data["source_sample_id"])
        c = conditions[gap].take(rows)
        z0 = arm.source(c, data, rows, step, calibrators[gap])
        target = arm.target(c, data, rows, args.device)
        model.train()
        optimizer.zero_grad()
        loss, _ = arm.objective(arm.bind(model, c), c, target, z0,
                               None if tau is None else tau.to(args.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        warmup = max(1, budget // 20)
        rate = step/warmup if step <= warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,budget-warmup)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * rate
        optimizer.step()
        if step % args.eval_every == 0 or step == budget:
            validate(step)
        if step % args.eval_every == 0 or step == budget or step == args.stop_after:
            state = checkpoint(step)
            state.update(optimizer=optimizer.state_dict(),
                         drivers={gap: dr.state_dict() for gap, dr in drivers.items()},
                         rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [])
            atomic_save(directory / "last.pt", state)
        if step == args.stop_after and step < budget:
            return dict(complete=False, stopped_step=step)
    if best is None:
        raise RuntimeError("No trained validation checkpoint selected")
    atomic_save(directory / "final.pt", checkpoint(budget))
    report = dict(complete=True, step=budget, selected_step=best["step"],
                  selected_sha256=sha(directory/"best.pt"), config=config,
                  test_arrays_read=False, engineering_smoke=args.smoke)
    save_json(directory / "complete.json", report)
    return report


def test_evaluate(args):
    configure()
    if args.checkpoint is None:
        raise ValueError("evaluate requires --checkpoint")
    if Path(args.output).exists():
        raise FileExistsError("Evaluation output must be new")
    checkpoint = Path(args.checkpoint)
    complete = json.loads(checkpoint.with_name("complete.json").read_text())
    if checkpoint.name != "best.pt" or sha(checkpoint) != complete["selected_sha256"]:
        raise ValueError("Evaluation requires a completed training checkpoint")
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    cfg = state["config"]
    if sha(args.data) != cfg["data_sha256"] or code_hash() != cfg["code_sha256"]:
        raise ValueError("Input or scientific source binding differs from training")
    if cfg["smoke"] and not args.allow_smoke:
        raise ValueError("Engineering smoke checkpoints require explicit --allow-smoke")
    name = args.arm or cfg["arm"]
    arm = Arm.resolve(name)
    selection_arm = Arm.resolve(cfg["arm"]).name
    if arm.name != selection_arm and not (
            arm.name == "Unconstrained-Flow" and selection_arm == "Terminal-Only"):
        raise ValueError("Only Unconstrained-Flow may reuse a Terminal-Only checkpoint")
    gaps = args.gaps or [cfg["gap"]]
    if any(gap != cfg["gap"] for gap in gaps):
        raise ValueError("Unregistered evaluation gap")
    directory = Path(args.output)
    directory.mkdir()
    # Freeze the selected weights before requesting any held-out arrays.
    save_json(directory/"selection_frozen_before_test.json", dict(
        checkpoint_sha256=sha(checkpoint), selected_step=state["step"], gaps=gaps,
        selection_arm=selection_arm, inference_arm=arm.name,
        independently_selected_unconstrained_flow=False))
    arrays, _, mean, std, meta = load_data(args.data, ("test",))
    model = arm.model(cfg["seed"], args.device)
    model.load_state_dict(state["model"], strict=True)
    report = {}
    for gap in gaps:
        data = d.make_data(arrays["test"], None, mean, std, gap)
        metrics, outputs = evaluate(model, arm, data, gap, args.device)
        np.savez_compressed(directory/f"mid{gap}_predictions.npz", **outputs)
        report[str(gap)] = metrics
    save_json(directory/"metrics.json", report)
    return report


def demo_data(path):
    """Synthetic engineering data; never part of a reported scientific result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    raw = {}
    for split, count in (("train", 20), ("val", 7), ("test", 9)):
        value = np.zeros((count, 25, 3), dtype=np.float64)
        value[:, :, :2] = rng.normal(size=(count,25,2)).cumsum(1)
        value[:, :, 2] = np.cumsum(rng.uniform(.2,1.1,(count,25)),axis=1)
        value[:, :, 2] -= value[:, :1, 2]
        raw[split] = value
    mean = raw["train"][:,:,:2].mean((0,1))
    scale = np.sqrt(raw["train"][:,:,:2].var((0,1)).mean())
    ts = raw["train"][:,:,2].std()
    meta = dict(dataset="atc", coordinates="B_normalized", mean_xy=mean.tolist(),
                xy_scale=[float(scale)]*2, time_scale=float(ts), normalization_fit="full_train",
                engineering_only=True, entity_disjoint=True, native_unit="synthetic")
    result = {"metadata": np.array(json.dumps(meta))}
    for split, value in raw.items():
        value[:,:,:2] = (value[:,:,:2]-mean)/scale
        value[:,:,2] /= ts
        result[split] = value.astype(np.float32)
        result[split+"_entity"] = np.array([split+"_"+str(i//2) for i in range(len(value))])
    np.savez_compressed(path, **result)


QUERY_PROTOCOL = {
    "positions": "Unnormalized [N,25,2] .npy; missing points are replaced by predictions",
    "times": "Original strictly increasing [N,25] .npy in seconds",
    "workload": {
        "spatial_lower": "[Q,2] inclusive lower box bounds",
        "spatial_upper": "[Q,2] inclusive upper box bounds",
        "time_start": "[Q] inclusive interval starts",
        "time_end": "[Q] inclusive interval ends",
        "knn_query_indices": "[K] uncorrupted query-window row indices",
        "stay_speed_threshold": "Positive training-derived speed threshold",
        "stay_min_duration": "Nonnegative training-derived duration threshold",
    },
    "metrics": ["range_F1", "knn_Recall_at_10", "stay_duration_MAE"],
    "normalization": "pred/truth are native physical coordinates; legacy normalization fields are optional and never applied",
}


def frechet_distances(query, candidates):
    coupling = np.full((len(query), len(query), len(candidates)), np.inf, dtype=np.float64)
    for row in range(len(query)):
        for column in range(len(query)):
            point_distance = np.linalg.norm(query[row] - candidates[:, column], axis=-1)
            if row == 0 and column == 0:
                coupling[row, column] = point_distance
            elif row == 0:
                coupling[row, column] = np.maximum(coupling[row, column - 1], point_distance)
            elif column == 0:
                coupling[row, column] = np.maximum(coupling[row - 1, column], point_distance)
            else:
                previous = np.minimum(coupling[row - 1, column], coupling[row - 1, column - 1])
                coupling[row, column] = np.maximum(np.minimum(previous, coupling[row, column - 1]), point_distance)
    return coupling[-1, -1]


def _workload_numeric(workload, key):
    try:
        return np.asarray(workload[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be numeric") from error


def evaluate_queries(args):
    with np.load(args.predictions, allow_pickle=False) as archive:
        if not {"pred", "truth"} <= set(archive.files):
            raise ValueError("Prediction archive must contain pred and truth")
        predicted = np.asarray(archive["pred"], dtype=np.float64)
        truth = np.asarray(archive["truth"], dtype=np.float64)
    positions = np.asarray(np.load(args.positions, allow_pickle=False), dtype=np.float64)
    times = np.asarray(np.load(args.times, allow_pickle=False), dtype=np.float64)
    workload = json.loads(Path(args.workload).read_text())
    if not isinstance(workload, dict):
        raise ValueError("Workload must be a JSON object")
    required = set(QUERY_PROTOCOL["workload"])
    if not required <= workload.keys():
        raise ValueError(f"Missing workload fields: {sorted(required - workload.keys())}")
    if positions.ndim != 3 or positions.shape[1:] != (25, 2) or not len(positions):
        raise ValueError("Expected nonempty positions [N,25,2]")
    if times.shape != positions.shape[:2]:
        raise ValueError("Expected times [N,25] matching positions")
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError("Positions and times must be finite")
    if not (np.diff(times, axis=1) > 0).all():
        raise ValueError("Times must be strictly increasing")
    if predicted.shape != truth.shape or predicted.shape != (len(positions), 13, 2):
        raise ValueError("Expected matching pred/truth [N,13,2] arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("pred/truth must be finite")
    # ``evaluate`` already stores native physical coordinates. Legacy workload
    # normalization fields are checked, but they are never applied a second time.
    for key in ("normalization_mean_xy", "normalization_scale_xy"):
        if key in workload:
            value = _workload_numeric(workload, key)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite [2] array")
            if key == "normalization_scale_xy" and not (value > 0).all():
                raise ValueError("normalization_scale_xy must be positive")
    repaired, uncorrupted = positions.copy(), positions.copy()
    repaired[:, 6:19], uncorrupted[:, 6:19] = predicted, truth
    lower = _workload_numeric(workload, "spatial_lower")
    upper = _workload_numeric(workload, "spatial_upper")
    start = _workload_numeric(workload, "time_start")
    end = _workload_numeric(workload, "time_end")
    if (lower.ndim != 2 or lower.shape[1] != 2 or lower.shape != upper.shape
            or not len(lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all()):
        raise ValueError("Range boxes must be finite matching [Q,2] bounds")
    if not (lower <= upper).all():
        raise ValueError("spatial_lower must not exceed spatial_upper")
    if (start.shape != (len(lower),) or end.shape != (len(lower),)
            or not np.isfinite(start).all() or not np.isfinite(end).all()):
        raise ValueError("time_start and time_end must be finite [Q] arrays")
    if not (start <= end).all():
        raise ValueError("time_start must not exceed time_end")
    raw_rows = np.asarray(workload["knn_query_indices"])
    if raw_rows.ndim != 1 or not len(raw_rows):
        raise ValueError("knn_query_indices must be a nonempty 1-D array")
    if not np.issubdtype(raw_rows.dtype, np.number) or not np.isfinite(raw_rows).all():
        raise ValueError("knn_query_indices must contain finite integers")
    if not np.equal(raw_rows, np.floor(raw_rows)).all():
        raise ValueError("knn_query_indices must contain integers")
    query_rows = raw_rows.astype(np.int64)
    if (query_rows < 0).any() or (query_rows >= len(positions)).any():
        raise ValueError("knn_query_indices must select existing rows")
    if len(positions) < 11:
        raise ValueError("kNN Recall@10 requires at least 11 trajectory windows")
    try:
        speed_threshold = float(workload["stay_speed_threshold"])
        min_duration = float(workload["stay_min_duration"])
    except (TypeError, ValueError) as error:
        raise ValueError("Stay thresholds must be scalar numbers") from error
    if not math.isfinite(speed_threshold) or speed_threshold <= 0:
        raise ValueError("stay_speed_threshold must be finite and positive")
    if not math.isfinite(min_duration) or min_duration < 0:
        raise ValueError("stay_min_duration must be finite and nonnegative")

    def range_answers(full):
        spatial = ((full[:, None, 6:19] >= lower[None, :, None]).all(-1)
                   & (full[:, None, 6:19] <= upper[None, :, None]).all(-1))
        temporal = ((times[:, None, 6:19] >= start[None, :, None])
                    & (times[:, None, 6:19] <= end[None, :, None]))
        return (spatial & temporal).any(-1).T

    actual, expected = range_answers(repaired), range_answers(uncorrupted)
    union = actual.sum(1) + expected.sum(1)
    overlap = (actual & expected).sum(1)
    scores = np.ones(len(lower), dtype=np.float64)
    np.divide(2 * overlap, union, out=scores, where=union != 0)
    distances = np.stack([frechet_distances(uncorrupted[row], repaired) for row in query_rows])
    distances[np.arange(len(query_rows)), query_rows] = np.inf
    predicted_neighbors = np.argsort(distances, axis=1, kind="stable")[:, :10]
    truth_distances = np.stack([frechet_distances(uncorrupted[row], uncorrupted) for row in query_rows])
    truth_distances[np.arange(len(query_rows)), query_rows] = np.inf
    expected_neighbors = np.argsort(truth_distances, axis=1, kind="stable")[:, :10]
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(predicted_neighbors, expected_neighbors)])

    def durations(full):
        slow = np.linalg.norm(np.diff(full, axis=1), axis=-1) / np.diff(times, axis=1) < speed_threshold
        result = np.zeros(len(full), dtype=np.float64)
        for row in range(len(full)):
            accumulated = 0.
            for slot in range(25):
                if slot < 24 and slow[row, slot]:
                    accumulated += times[row, slot + 1] - times[row, slot]
                else:
                    if accumulated >= min_duration:
                        result[row] += accumulated
                    accumulated = 0.
        return result

    report = dict(range_F1=float(scores.mean()), knn_Recall_at_10=float(recall),
                  stay_duration_MAE=float(np.abs(durations(repaired) - durations(uncorrupted)).mean()),
                  range_queries=len(lower), knn_queries=len(query_rows), windows=len(positions))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "query_metrics.json", report)
    return report


def paper_arm_name(value):
    """Canonicalize compatibility aliases while exposing only paper variants."""
    try:
        return Arm.resolve(value).name
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-data")
    demo.add_argument("--output", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--dataset", required=True, choices=d.DATASETS)
    tr.add_argument("--arm", default="TanFlow-Full", type=paper_arm_name, choices=ARMS)
    tr.add_argument("--gap", type=int, choices=(5,9,13,17,21), default=13)
    tr.add_argument("--seed", type=int, choices=(0,1,3), default=0)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--steps", type=int)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--stop-after", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", help="Validation-best checkpoint")
    ev.add_argument("--output", required=True)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--arm", type=paper_arm_name, choices=ARMS)
    ev.add_argument("--gaps", type=int, nargs="+",
                    help="Centered gaps; defaults to the checkpoint config")
    ev.add_argument("--allow-smoke", action="store_true")
    qp = sub.add_parser("query-protocol", description="Print downstream query schema")
    qq = sub.add_parser("evaluate-queries", description="Evaluate downstream spatial-temporal queries")
    qq.add_argument("--predictions", required=True)
    qq.add_argument("--positions", required=True)
    qq.add_argument("--times", required=True)
    qq.add_argument("--workload", required=True)
    qq.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "demo-data":
        demo_data(args.output)
        print("Synthetic data created; not a scientific result.")
    elif args.command == "query-protocol":
        print(json.dumps(QUERY_PROTOCOL, indent=2))
    elif args.command == "evaluate-queries":
        print(json.dumps(evaluate_queries(args), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(test_evaluate(args), indent=2))
    else:
        print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()

"""Uncentered second-moment statistics over training velocity intervals."""
import torch

def _synthetic_sigma(residual, active):
    # Per-interval second moment of V_K residuals across the WHOLE train split,
    # broadcast to every window.  Statistics are global per interval index and
    # never read the querying window's own ground-truth residual, so no
    # per-window leakage is introduced; val/test reuse the train-fitted tensor.
    steps = residual.shape[1]
    act = active[0].bool()
    sigma = torch.zeros(steps, 2, 2, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    for j in range(steps):
        if act[j]:
            r = residual[:, j, :]
            sigma[j] = (r.T @ r) / max(len(r) - 1, 1)
    iso = sum(float(sigma[j].trace()) / 2 for j in range(steps) if act[j]) / max(int(act.sum()), 1)
    for j in range(steps):
        if not act[j]:
            sigma[j] = max(iso, 1e-12) * eye
    return sigma, iso


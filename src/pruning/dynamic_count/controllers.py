"""
controllers.py — Dynamic-COUNT controllers (pure CPU; fitted ONLY on the calibration split).

Budget semantics are FRACTIONS of the sample's native visual token count N_i, so one controller
definition serves both models: LLaVA N_i = 576 (fixed), Qwen N_i = native per-image count. The
executed integer budget is K_i = clamp(round(fraction · N_i), MIN_TOKENS, N_i) — a continuum
(64, 97, 188, 240, 391, …), never restricted to the anchors.

  * RuleController (PRIMARY, transparent): risk-quantile bins → per-bin sufficiency fraction read
    off the anchor accuracy curve (linear interpolation BETWEEN anchors → continuous targets),
    isotonic-smoothed so higher risk never gets fewer tokens; piecewise-linear risk→fraction at
    predict time; compute knob λ scales all targets (sweeps the accuracy-vs-FLOPs curve).
  * RidgeRiskModel (SECONDARY): tiny closed-form ridge regression on the standardized signal
    vector → calibrated risk in [0,1]; that risk feeds the SAME rule mapping. ≤ ~30 weights,
    thesis-explainable, no backbone training.
  * Discrete cascade helpers (DC-D): risk threshold τ chosen on calibration (by escalation-rate
    quantile), escalate → a fixed anchor.

All fitted state serializes via to_dict()/from_dict() (JSON round-trip) so calibration and
execution are decoupled scripts.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from src.pruning.dynamic_count import ANCHOR_FRACTIONS


# ── shared pure helpers ────────────────────────────────────────────────────────

def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation, clamped at both ends. xs must be ascending."""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 + 1e-12)
    return ys[-1]


def sufficiency_fraction(anchor_accs: Dict[float, float], eps: float = 0.01) -> float:
    """Smallest fraction whose accuracy reaches (max accuracy − eps), linearly interpolated
    BETWEEN anchors → continuous target. anchor_accs: {fraction: mean score in [0,1]}."""
    fr = sorted(anchor_accs)
    accs = [anchor_accs[f] for f in fr]
    thr = max(accs) - eps
    for j, a in enumerate(accs):
        if a >= thr:
            if j == 0:
                return fr[0]
            a0, a1 = accs[j - 1], accs[j]
            if a1 - a0 <= 1e-12:
                return fr[j]
            return fr[j - 1] + (fr[j] - fr[j - 1]) * (thr - a0) / (a1 - a0)
    return fr[-1]


def _gauss_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting (A: small ridge normal matrix,
    symmetric positive-definite thanks to the +alpha ridge, so this is numerically safe)."""
    k = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-12:
            continue
        M[col] = [v / p for v in M[col]]
        for r in range(k):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    return [M[i][k] for i in range(k)]


def threshold_for_rate(risks: Sequence[float], rate: float) -> float:
    """τ such that ~`rate` of samples have risk > τ (escalate). rate in [0,1]."""
    if not risks:
        return float("inf")
    s = sorted(risks)
    idx = int(round((1.0 - max(0.0, min(1.0, rate))) * (len(s) - 1)))
    return s[max(0, min(idx, len(s) - 1))]


# ── PRIMARY: transparent rule controller (risk → continuous budget fraction) ──

class RuleController:
    """Risk-quantile bins → interpolated sufficiency fractions (isotonic-smoothed)."""

    def __init__(self, nbins: int = 8, eps: float = 0.01):
        self.nbins = int(nbins)
        self.eps = float(eps)
        self.bin_centers: List[float] = []
        self.bin_targets: List[float] = []

    def fit(self, risks: Sequence[float], anchor_scores: Sequence[Dict[float, float]]) -> "RuleController":
        """risks[i] = calibration sample i's risk; anchor_scores[i] = {fraction: score} for the
        SAME sample at every anchor fraction (incl. 1.0 = dense), read from frozen finals."""
        n = len(risks)
        if n == 0:
            raise ValueError("no calibration samples")
        order = sorted(range(n), key=lambda i: risks[i])
        nb = max(1, min(self.nbins, n))
        centers, targets = [], []
        for b in range(nb):
            mem = order[b * n // nb:(b + 1) * n // nb]
            if not mem:
                continue
            accs = {f: sum(anchor_scores[i][f] for i in mem) / len(mem) for f in ANCHOR_FRACTIONS}
            centers.append(sum(risks[i] for i in mem) / len(mem))
            targets.append(sufficiency_fraction(accs, self.eps))
        # isotonic smoothing: higher risk must never get FEWER tokens
        run = 0.0
        targets = [run := max(run, t) for t in targets]
        self.bin_centers, self.bin_targets = centers, targets
        return self

    def predict_fraction(self, risk: float, lam: float = 1.0) -> float:
        """Continuous budget fraction for this risk (λ scales the whole allocation)."""
        if not self.bin_centers:
            raise RuntimeError("controller not fitted")
        f = _interp(self.bin_centers, self.bin_targets, float(risk))
        return max(0.0, min(1.0, lam * f))

    def to_dict(self) -> dict:
        return {"type": "rule", "nbins": self.nbins, "eps": self.eps,
                "bin_centers": self.bin_centers, "bin_targets": self.bin_targets}

    @classmethod
    def from_dict(cls, d: dict) -> "RuleController":
        c = cls(nbins=d["nbins"], eps=d["eps"])
        c.bin_centers = list(d["bin_centers"])
        c.bin_targets = list(d["bin_targets"])
        return c


# ── SECONDARY: tiny calibrated ridge regression → risk in [0,1] ────────────────

class RidgeRiskModel:
    """Closed-form ridge on standardized features; target = 1(wrong at probe). Output clipped to
    [0,1] → a calibrated risk that feeds the same RuleController mapping."""

    def __init__(self, feature_names: Sequence[str], alpha: float = 1.0):
        self.feature_names = list(feature_names)
        self.alpha = float(alpha)
        self.mu: List[float] = []
        self.sd: List[float] = []
        self.w: List[float] = []
        self.b: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "RidgeRiskModel":
        # pure-python closed-form ridge (k ≤ ~30 features → tiny Gaussian solve; no numpy needed
        # so the controller stays importable/testable under the plain CPU python)
        n = len(X)
        k = len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(k)]
        self.sd = [max((sum((r[j] - self.mu[j]) ** 2 for r in X) / n) ** 0.5, 1e-8) for j in range(k)]
        Xs = [[(r[j] - self.mu[j]) / self.sd[j] for j in range(k)] for r in X]
        ybar = sum(y) / n
        yc = [v - ybar for v in y]
        A = [[sum(Xs[i][a] * Xs[i][b] for i in range(n)) + (self.alpha if a == b else 0.0)
              for b in range(k)] for a in range(k)]
        rhs = [sum(Xs[i][a] * yc[i] for i in range(n)) for a in range(k)]
        self.w = _gauss_solve(A, rhs)
        self.b = float(ybar)
        return self

    def predict_risk(self, x: Sequence[float]) -> float:
        z = sum((float(v) - m) / s * w for v, m, s, w in zip(x, self.mu, self.sd, self.w)) + self.b
        return max(0.0, min(1.0, z))

    def to_dict(self) -> dict:
        return {"type": "ridge", "feature_names": self.feature_names, "alpha": self.alpha,
                "mu": self.mu, "sd": self.sd, "w": self.w, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "RidgeRiskModel":
        m = cls(d["feature_names"], alpha=d["alpha"])
        m.mu, m.sd, m.w, m.b = list(d["mu"]), list(d["sd"]), list(d["w"]), float(d["b"])
        return m


def load_controller(d: dict):
    """Deserialize any controller payload written by the calibration script."""
    if d.get("type") == "rule":
        return RuleController.from_dict(d)
    if d.get("type") == "ridge":
        return RidgeRiskModel.from_dict(d)
    raise ValueError(f"unknown controller type {d.get('type')!r}")

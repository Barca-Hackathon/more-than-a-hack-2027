"""A two-state hidden Markov model with Gaussian and Bernoulli emissions, fitted by EM.

Used by deadball.py for live (state 0) vs dead (state 1). Generic and dependency-free:

  * emissions are conditionally independent given the state: diagonal Gaussians on the columns
    of G and Bernoullis on the columns of B; NaN in either means "not observed", contributing
    nothing to that frame's likelihood;
  * frames can be clamped to a state (anchors), which pins the meaning of the states;
  * several independent sequences (the two halves) share one set of parameters.

The recursions are written for exactly two states with plain Python floats. That is ~10x faster
than numpy for K=2, where per-call overhead dominates, and keeps the dependency list empty.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

SD_FLOOR = 0.05
P_FLOOR = 0.01
TRANS_FLOOR = 1e-9


@dataclass(frozen=True)
class Params:
    mu: np.ndarray       # (2, G) Gaussian means
    sd: np.ndarray       # (2, G) Gaussian standard deviations
    p: np.ndarray        # (2, B) Bernoulli probabilities
    p01: float           # P(state 1 at t+1 | state 0 at t)
    p10: float           # P(state 0 at t+1 | state 1 at t)
    init1: float = 0.5   # P(state 1 at the start of a sequence)

    def at_stride(self, stride):
        """Per-frame transition probabilities of a model fitted on every `stride`-th frame."""
        if stride == 1:
            return self
        convert = lambda q: 1.0 - (1.0 - q) ** (1.0 / stride)  # noqa: E731
        return replace(self, p01=convert(self.p01), p10=convert(self.p10))


def emission_loglik(params, G, B):
    """(n, 2) log-likelihood of each frame's observations under each state."""
    n = len(G) if G is not None else len(B)
    out = np.zeros((n, 2))
    if G is not None and G.shape[1]:
        observed = ~np.isnan(G)
        values = np.where(observed, G, 0.0)
        for state in (0, 1):
            z = (values - params.mu[state]) / params.sd[state]
            term = -0.5 * z * z - np.log(params.sd[state]) - 0.5 * math.log(2 * math.pi)
            out[:, state] += np.where(observed, term, 0.0).sum(axis=1)
    if B is not None and B.shape[1]:
        observed = ~np.isnan(B)
        values = np.where(observed, B, 0.0)
        for state in (0, 1):
            p = params.p[state]
            term = values * np.log(p) + (1.0 - values) * np.log1p(-p)
            out[:, state] += np.where(observed, term, 0.0).sum(axis=1)
    return out


def forward_backward(loglik, p01, p10, init1=0.5, clamp1=None, clamp0=None):
    """Posterior P(state 1) per frame for one sequence, plus expected transition counts.

    Returns (gamma1, counts, total_loglik) with counts = (n00, n01, n10, n11).
    """
    loglik = np.asarray(loglik, dtype=np.float64)
    n = len(loglik)
    if n == 0:
        return np.zeros(0), (0.0, 0.0, 0.0, 0.0), 0.0
    ll0 = loglik[:, 0].copy()
    ll1 = loglik[:, 1].copy()
    scale = np.maximum(ll0, ll1)
    if clamp1 is not None:
        clamp1 = np.asarray(clamp1, dtype=bool)
        scale = np.where(clamp1, ll1, scale)
    if clamp0 is not None:
        clamp0 = np.asarray(clamp0, dtype=bool)
        scale = np.where(clamp0, ll0, scale)
    l0 = np.exp(ll0 - scale)
    l1 = np.exp(ll1 - scale)
    if clamp1 is not None:
        l0[clamp1] = 0.0
    if clamp0 is not None:
        l1[clamp0] = 0.0
    l0, l1 = l0.tolist(), l1.tolist()

    a01 = min(max(float(p01), TRANS_FLOOR), 1 - TRANS_FLOOR)
    a10 = min(max(float(p10), TRANS_FLOOR), 1 - TRANS_FLOOR)
    s00, s11 = 1.0 - a01, 1.0 - a10

    al0 = [0.0] * n
    al1 = [0.0] * n
    cs = [0.0] * n
    x0 = (1.0 - init1) * l0[0]
    x1 = init1 * l1[0]
    c = x0 + x1 or 1e-300
    al0[0], al1[0], cs[0] = x0 / c, x1 / c, c
    for t in range(1, n):
        prev0, prev1 = al0[t - 1], al1[t - 1]
        x0 = (prev0 * s00 + prev1 * a10) * l0[t]
        x1 = (prev0 * a01 + prev1 * s11) * l1[t]
        c = x0 + x1 or 1e-300
        al0[t], al1[t], cs[t] = x0 / c, x1 / c, c

    be0 = [1.0] * n
    be1 = [1.0] * n
    n00 = n01 = n10 = n11 = 0.0
    for t in range(n - 1, 0, -1):
        c = cs[t]
        y0 = l0[t] * be0[t] / c
        y1 = l1[t] * be1[t] / c
        prev0, prev1 = al0[t - 1], al1[t - 1]
        n00 += prev0 * s00 * y0
        n01 += prev0 * a01 * y1
        n10 += prev1 * a10 * y0
        n11 += prev1 * s11 * y1
        be0[t - 1] = s00 * y0 + a01 * y1
        be1[t - 1] = a10 * y0 + s11 * y1

    g0 = np.asarray(al0) * np.asarray(be0)
    g1 = np.asarray(al1) * np.asarray(be1)
    total = g0 + g1
    total[total == 0] = 1.0
    loglik_total = float(np.sum(np.log(cs)) + np.sum(scale))
    return g1 / total, (n00, n01, n10, n11), loglik_total


def m_step(G, B, gamma1, counts, params):
    """Re-estimate every parameter from posteriors. Floors keep it away from degeneracy."""
    weights = np.stack([1.0 - gamma1, gamma1])          # (2, n)
    mu, sd, p = params.mu.copy(), params.sd.copy(), params.p.copy()
    if G is not None and G.shape[1]:
        observed = ~np.isnan(G)
        values = np.where(observed, G, 0.0)
        for state in (0, 1):
            w = weights[state][:, None] * observed
            mass = np.maximum(w.sum(axis=0), 1e-9)
            mu[state] = (w * values).sum(axis=0) / mass
            var = (w * (values - mu[state]) ** 2).sum(axis=0) / mass
            sd[state] = np.maximum(np.sqrt(var), SD_FLOOR)
    if B is not None and B.shape[1]:
        observed = ~np.isnan(B)
        values = np.where(observed, B, 0.0)
        for state in (0, 1):
            w = weights[state][:, None] * observed
            mass = np.maximum(w.sum(axis=0), 1e-9)
            p[state] = np.clip((w * values).sum(axis=0) / mass, P_FLOOR, 1 - P_FLOOR)
    n00, n01, n10, n11 = counts
    p01 = n01 / max(n00 + n01, 1e-9)
    p10 = n10 / max(n10 + n11, 1e-9)
    return Params(mu=mu, sd=sd, p=p, p01=float(p01), p10=float(p10), init1=params.init1)


def posterior(params, G, B, sequences, clamp1=None, clamp0=None, temperature=1.0):
    """(gamma1 over the full arrays, NaN outside sequences; counts; total log-likelihood).

    temperature divides the emission log-likelihoods. A model fitted on every k-th frame and
    evaluated on every frame must use temperature=k, or each observation's evidence is counted
    k times and the posterior becomes overconfident.
    """
    n = len(G) if G is not None else len(B)
    loglik = emission_loglik(params, G, B) / float(temperature)
    gamma = np.full(n, np.nan)
    counts = np.zeros(4)
    total = 0.0
    for start, stop in sequences:
        part = slice(start, stop)
        g, c, ll = forward_backward(
            loglik[part], params.p01, params.p10, params.init1,
            None if clamp1 is None else clamp1[part],
            None if clamp0 is None else clamp0[part])
        gamma[part] = g
        counts += c
        total += ll
    return gamma, tuple(counts), total


def fit(G, B, sequences, params, clamp1=None, clamp0=None, n_iter=60, tol=1e-6, stride=1):
    """EM on every `stride`-th frame of each sequence. Returns (params at frame rate, history).

    Fitting on a stride is a speed trick that costs nothing here: the features are rolling-window
    aggregates, so neighbouring frames carry almost no extra information. The transition
    probabilities are converted back to the per-frame rate before returning.
    """
    pick = np.concatenate([np.arange(a, b, stride) for a, b in sequences]) if sequences else []
    pick = np.asarray(pick, dtype=np.int64)
    Gs = None if G is None else G[pick]
    Bs = None if B is None else B[pick]
    c1 = None if clamp1 is None else np.asarray(clamp1, bool)[pick]
    c0 = None if clamp0 is None else np.asarray(clamp0, bool)[pick]
    # Strided sequences, re-indexed into the compact arrays.
    sub, offset = [], 0
    for a, b in sequences:
        length = len(range(a, b, stride))
        sub.append((offset, offset + length))
        offset += length

    history = []
    for _ in range(n_iter):
        gamma, counts, total = posterior(params, Gs, Bs, sub, c1, c0)
        history.append(total)
        params = m_step(Gs, Bs, gamma, counts, params)
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * abs(history[-2]):
            break
    return params.at_stride(stride), history

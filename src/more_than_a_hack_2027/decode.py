"""Segment decoding: per-frame evidence for {H, A, D} -> a piecewise-constant label sequence.

This is where spellF1 is won or lost. The metric scores possessions as whole intervals, so a
one-frame flicker costs two spurious possessions while barely moving macroF1. Instead of voting
frame by frame, we find the label sequence that maximises

    sum_t evidence[t, label_t]  +  sum_t transition[label_{t-1}, label_t]

with Viterbi. The transition scores are non-positive penalties (0 for staying), so this is an
energy model rather than a normalised HMM -- penalties are the natural knob and need no
normalisation. Decoding runs inside each half separately, never across halftime.

Penalties are hyperparameters: choose them on the external validation games, never on the
public five minutes (RULES.md, EVALUATION.md).
"""

import numpy as np

H, A, D = 0, 1, 2
STATES = ("H", "A", "D")
N_STATES = len(STATES)
UNGRADED_LABEL = "X"   # outside the halves nothing is scored, so we abstain honestly there


def transition_matrix(switch, stop, restart):
    """(3, 3) log-score of moving from row state to column state.

    switch   H <-> A, a turnover in open play
    stop     H/A -> D, the ball going dead
    restart  D -> H/A, play resuming
    """
    for name, value in (("switch", switch), ("stop", stop), ("restart", restart)):
        if value < 0:
            raise ValueError("{} penalty must be >= 0, got {}".format(name, value))
    trans = np.zeros((N_STATES, N_STATES))
    trans[H, A] = trans[A, H] = -float(switch)
    trans[H, D] = trans[A, D] = -float(stop)
    trans[D, H] = trans[D, A] = -float(restart)
    return trans


def viterbi(evidence, trans, initial=None):
    """Best state path for one contiguous block. evidence is (n, K) log-scores, finite.

    Ties are broken towards the lower state index (H, then A, then D) at every step, so the
    result is deterministic.
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    n, k = evidence.shape
    if trans.shape != (k, k):
        raise ValueError("transition matrix {} does not match {} states".format(trans.shape, k))
    if not np.isfinite(evidence).all():
        raise ValueError("evidence must be finite; map 'no evidence' to 0, not NaN")
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    back = np.empty((n, k), dtype=np.int8)
    score = evidence[0] + (0.0 if initial is None else np.asarray(initial, dtype=np.float64))
    for t in range(1, n):
        candidates = score[:, None] + trans          # (from, to)
        best_from = candidates.argmax(axis=0)
        back[t] = best_from
        score = candidates[best_from, np.arange(k)] + evidence[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_dead(evidence, trans, min_dead):
    """Like viterbi over (H, A, D), but every D run inside the block lasts >= min_dead frames.

    A run of D that starts the block is exempt (a half can open dead for any length). Without
    this, a turnover in open play can be bought as H -> D -> A through a single frame for
    stop + restart + one frame of dead evidence, which silently caps the switch penalty.

    The D chain is deterministic, so only three running scores are needed:
        H[t]  = e_H[t] + max(H[t-1], A[t-1] - switch, D[t-1] - restart)          (A alike)
        D[t]  = max(D[t-1] + e_D[t],                      stay in a full-length run
                    max(H, A)[t-m] - stop + E_D(t-m, t])  enter m frames ago
    where E_D sums the D evidence over the m newest frames. Ties go to the lower state index
    and to staying, as in viterbi().
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    n = len(evidence)
    if evidence.shape[1:] != (N_STATES,):
        raise ValueError("viterbi_min_dead decodes exactly (H, A, D)")
    if not np.isfinite(evidence).all():
        raise ValueError("evidence must be finite; map 'no evidence' to 0, not NaN")
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    m = max(1, int(min_dead))
    switch, stop, restart = -trans[H, A], -trans[H, D], -trans[D, H]
    if trans[A, H] != trans[H, A] or trans[A, D] != trans[H, D] or trans[D, A] != trans[D, H]:
        raise ValueError("viterbi_min_dead expects the symmetric transition_matrix()")

    e_h = evidence[:, H].tolist()
    e_a = evidence[:, A].tolist()
    e_d = evidence[:, D]
    cum = np.concatenate(([0.0], np.cumsum(e_d))).tolist()      # cum[t + 1] = sum e_d[0..t]
    e_d = e_d.tolist()
    neg = -np.inf

    sh, sa, sd = [0.0] * n, [0.0] * n, [0.0] * n
    bh, ba = bytearray(n), bytearray(n)       # 0 = from H, 1 = from A, 2 = from D
    bd = bytearray(n)                         # 0 = stayed in D, 1 = entered m frames ago
    enter_from = bytearray(n)                 # for an entry ending at t: 0 = from H, 1 = from A
    sh[0], sa[0], sd[0] = e_h[0], e_a[0], e_d[0]
    for t in range(1, n):
        ph, pa, pd = sh[t - 1], sa[t - 1], sd[t - 1]
        best, arg = ph, 0
        if pa - switch > best:
            best, arg = pa - switch, 1
        if pd - restart > best:
            best, arg = pd - restart, 2
        sh[t], bh[t] = best + e_h[t], arg
        best, arg = pa, 1
        if ph - switch > best:
            best, arg = ph - switch, 0
        if pd - restart > best:
            best, arg = pd - restart, 2
        sa[t], ba[t] = best + e_a[t], arg
        stay = pd + e_d[t]
        start = t - m + 1                       # first frame of a run that reaches m frames at t
        if start >= 1:
            qh, qa = sh[start - 1], sa[start - 1]
            source, came = (qh, 0) if qh >= qa else (qa, 1)
            entry = source - stop + cum[t + 1] - cum[start]
        else:
            entry, came = neg, 0
        if entry > stay:
            sd[t], bd[t], enter_from[t] = entry, 1, came
        else:
            sd[t], bd[t] = stay, 0

    path = np.empty(n, dtype=np.int8)
    t = n - 1
    final = (sh[t], sa[t], sd[t])
    state = int(np.argmax(final))
    while t >= 0:
        if state == D:
            if t > 0 and bd[t] == 1:
                start = t - m + 1
                path[start:t + 1] = D
                state, t = int(enter_from[t]), start - 1
                continue
            path[t] = D
            t -= 1
            continue
        path[t] = state
        if t == 0:
            break
        state = int((bh if state == H else ba)[t])
        t -= 1
    return path


def decode_match(evidence, halves, trans, initial=None, min_dead=1):
    """Full-length label array: Viterbi inside each half, UNGRADED_LABEL elsewhere.

    halves is ctx.halves: ((half, first_frame, last_frame), ...) with inclusive bounds.
    min_dead > 1 forbids D runs shorter than that many frames (viterbi_min_dead).
    """
    evidence = np.asarray(evidence, dtype=np.float64)
    labels = np.full(len(evidence), UNGRADED_LABEL, dtype="<U1")
    names = np.array(STATES)
    for _, first, last in halves:
        block = evidence[first:last + 1]
        if min_dead > 1:
            if initial is not None:
                raise ValueError("initial scores are not supported with min_dead > 1")
            path = viterbi_min_dead(block, trans, min_dead)
        else:
            path = viterbi(block, trans, initial)
        labels[first:last + 1] = names[path]
    return labels

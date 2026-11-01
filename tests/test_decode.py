"""The decoder on synthetic evidence -- no match data needed."""

import numpy as np
import pytest

from more_than_a_hack_2027.stages import decode
from more_than_a_hack_2027.stages.decode import A, D, H


def evidence_for(states, strength=1.0, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    ev = rng.normal(0.0, noise, size=(len(states), 3)) if noise else np.zeros((len(states), 3))
    ev[np.arange(len(states)), states] += strength
    return ev


def test_zero_penalties_is_framewise_argmax():
    ev = np.random.RandomState(1).normal(size=(500, 3))
    path = decode.viterbi(ev, decode.transition_matrix(0, 0, 0))
    assert np.array_equal(path, ev.argmax(axis=1))


def test_penalties_remove_flicker_and_keep_real_segments():
    truth = np.array([H] * 200 + [A] * 150 + [D] * 100 + [A] * 200)
    flicker = truth.copy()
    flicker[[50, 51, 260, 500]] = [A, A, H, D]          # isolated wrong frames
    ev = evidence_for(flicker, strength=1.0, noise=0.3)
    path = decode.viterbi(ev, decode.transition_matrix(switch=8, stop=8, restart=8))
    assert np.array_equal(path, truth)


def test_short_but_strong_segment_survives():
    # Sub-second possessions are real in the ground truth; strong evidence must be able to win.
    truth = np.array([A] * 300 + [H] * 20 + [A] * 300)
    path = decode.viterbi(evidence_for(truth, strength=2.0),
                          decode.transition_matrix(switch=10, stop=20, restart=20))
    assert np.array_equal(path, truth)


def test_huge_penalty_gives_one_state():
    truth = np.array([H] * 100 + [A] * 100)
    path = decode.viterbi(evidence_for(truth), decode.transition_matrix(1e6, 1e6, 1e6))
    assert len(set(path.tolist())) == 1


def test_decode_match_never_crosses_halves_and_abstains_outside():
    n = 1000
    halves = ((1, 100, 399), (2, 600, 899))
    states = np.full(n, D)
    states[100:400] = H
    states[600:900] = A
    labels = decode.decode_match(evidence_for(states), halves, decode.transition_matrix(5, 5, 5))
    assert set(labels[:100]) == {"X"} and set(labels[400:600]) == {"X"}
    assert set(labels[100:400]) == {"H"} and set(labels[600:900]) == {"A"}


def test_rejects_nan_and_negative_penalties():
    with pytest.raises(ValueError):
        decode.viterbi(np.full((3, 3), np.nan), decode.transition_matrix(1, 1, 1))
    with pytest.raises(ValueError):
        decode.transition_matrix(-1, 0, 0)


def _brute_min_dead(evidence, trans, m):
    import itertools
    best, best_path = -np.inf, None
    n = len(evidence)
    for path in itertools.product(range(3), repeat=n):
        ok = True
        k = 0
        while k < n:                                   # every D run not at the start has >= m
            if path[k] == decode.D:
                j = k
                while j < n and path[j] == decode.D:
                    j += 1
                if k > 0 and j - k < m:
                    ok = False
                    break
                k = j
            else:
                k += 1
        if not ok:
            continue
        score = sum(evidence[t, s] for t, s in enumerate(path))
        score += sum(trans[a, b] for a, b in zip(path, path[1:]))
        if score > best + 1e-12:
            best, best_path = score, path
    return best, best_path


def _score(evidence, trans, path):
    return (sum(evidence[t, s] for t, s in enumerate(path))
            + sum(trans[a, b] for a, b in zip(path, path[1:])))


def test_min_dead_viterbi_matches_constrained_enumeration():
    rng = np.random.RandomState(7)
    trans = decode.transition_matrix(2.0, 1.0, 1.5)
    for m in (1, 2, 3):
        for _ in range(15):
            evidence = rng.normal(size=(7, 3)) * 2
            best, _ = _brute_min_dead(evidence, trans, m)
            path = decode.viterbi_min_dead(evidence, trans, m)
            assert np.isclose(_score(evidence, trans, path), best)


def test_min_dead_one_equals_plain_viterbi_score():
    rng = np.random.RandomState(8)
    trans = decode.transition_matrix(3.0, 1.0, 1.0)
    evidence = rng.normal(size=(300, 3))
    a = decode.viterbi(evidence, trans)
    b = decode.viterbi_min_dead(evidence, trans, 1)
    assert np.isclose(_score(evidence, trans, a), _score(evidence, trans, b))


def test_min_dead_closes_the_one_frame_turnover_shortcut():
    n = 60
    evidence = np.zeros((n, 3))
    evidence[:, decode.D] = -6.0
    evidence[:30, decode.H] = 0.0
    evidence[:30, decode.A] = -0.4
    evidence[30:, decode.H] = -0.4
    evidence[30:, decode.A] = 0.0
    trans = decode.transition_matrix(25.0, 2.0, 2.0)     # switching costs more than a D detour
    plain = decode.viterbi(evidence, trans)
    assert decode.D in plain                            # bought the turnover through D
    strict = decode.viterbi_min_dead(evidence, trans, 25)
    assert decode.D not in strict

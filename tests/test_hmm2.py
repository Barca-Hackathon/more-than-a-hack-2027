"""The two-state HMM against brute force, and EM parameter recovery."""

import itertools

import numpy as np

from more_than_a_hack_2027 import hmm2


def brute_force(loglik, p01, p10, init1, clamp1=None):
    n = len(loglik)
    trans = np.array([[1 - p01, p01], [p10, 1 - p10]])
    weights, posts = [], np.zeros(n)
    for path in itertools.product((0, 1), repeat=n):
        if clamp1 is not None and any(c and s == 0 for c, s in zip(clamp1, path)):
            continue
        logw = np.log(init1 if path[0] else 1 - init1) + loglik[0, path[0]]
        for t in range(1, n):
            logw += np.log(trans[path[t - 1], path[t]]) + loglik[t, path[t]]
        weights.append(np.exp(logw))
        posts += np.exp(logw) * np.array(path)
    total = np.sum(weights)
    return posts / total, np.log(total)


def test_forward_backward_matches_enumeration():
    rng = np.random.RandomState(3)
    loglik = rng.normal(size=(7, 2)) * 2
    gamma, _, total = hmm2.forward_backward(loglik, 0.2, 0.35, 0.4)
    expected, expected_total = brute_force(loglik, 0.2, 0.35, 0.4)
    assert np.allclose(gamma, expected)
    assert np.isclose(total, expected_total)


def test_clamping_matches_constrained_enumeration():
    rng = np.random.RandomState(4)
    loglik = rng.normal(size=(6, 2)) * 2
    clamp = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    gamma, _, total = hmm2.forward_backward(loglik, 0.1, 0.3, 0.5, clamp1=clamp)
    expected, expected_total = brute_force(loglik, 0.1, 0.3, 0.5, clamp1=clamp)
    assert np.allclose(gamma, expected)
    assert np.isclose(total, expected_total)
    assert np.all(gamma[clamp] == 1.0)


def test_em_recovers_parameters():
    rng = np.random.RandomState(0)
    n = 20000
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        flip = rng.rand() < (0.01 if states[t - 1] == 0 else 0.02)
        states[t] = 1 - states[t - 1] if flip else states[t - 1]
    mu = np.array([[1.0], [-1.0]])
    G = rng.normal(mu[states, 0], 0.5)[:, None]
    B = (rng.rand(n) < np.where(states == 1, 0.2, 0.8)).astype(float)[:, None]
    G[rng.rand(n) < 0.1] = np.nan                     # missing observations are allowed

    start = hmm2.Params(mu=np.array([[0.5], [-0.5]]), sd=np.ones((2, 1)),
                        p=np.array([[0.6], [0.4]]), p01=0.05, p10=0.05)
    params, history = hmm2.fit(G, B, [(0, n)], start, n_iter=100)
    assert np.allclose(params.mu[:, 0], [1.0, -1.0], atol=0.05)
    assert np.allclose(params.p[:, 0], [0.8, 0.2], atol=0.03)
    assert abs(params.p01 - 0.01) < 0.004 and abs(params.p10 - 0.02) < 0.006
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:]))   # EM never decreases


def test_temperature_softens_the_posterior():
    rng = np.random.RandomState(1)
    G = rng.normal(size=(500, 1))
    params = hmm2.Params(mu=np.array([[0.3], [-0.3]]), sd=np.ones((2, 1)),
                         p=np.zeros((2, 0)), p01=0.01, p10=0.01)
    sharp, _, _ = hmm2.posterior(params, G, None, [(0, 500)])
    soft, _, _ = hmm2.posterior(params, G, None, [(0, 500)], temperature=5.0)
    assert np.mean(np.abs(soft - 0.5)) < np.mean(np.abs(sharp - 0.5))

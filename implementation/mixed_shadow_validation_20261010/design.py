"""Fresh mixed-type samples and a finite, class-specific population target."""
from __future__ import annotations

import hashlib
from itertools import product
from math import factorial
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
from ads_shadow_mixed_preflight_20261010 import adapter
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import softmax

SCENARIOS = ("additive", "rare_interaction", "null")
MASTER = 2026101001
FEATURES = [f"X{i}" for i in range(1, 11)]
NUMERIC = ["X1", "X3", "X4", "X8"]


def specification(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    levels = [(-1., 0., 1.), ("a", "b", "c"), (-1., 1.),
              tuple(np.linspace(-1, 1, 5)), tuple(f"c{i}" for i in range(12)),
              ("common", "less_common", "rare"), ("off", "on"),
              tuple(np.linspace(-1, 1, 7)), tuple(f"v{i}" for i in range(8)),
              ("a", "b", "c", "d")]
    masses = [[.25, .5, .25], [.7, .2, .1], [.6, .4], [.2]*5, [1/12]*12,
              [.95, .04, .01], [.5, .5], [1/7]*7,
              [.65, .15, .08, .05, .03, .02, .015, .005], [.4, .3, .2, .1]]
    prior = [.8, .15, .05]
    if scenario == "rare_interaction":
        masses[1] = [.9, .09, .01]
        prior = [.9, .08, .02]
    return levels, [np.asarray(v, float) for v in masses], np.array(prior)


def population(scenario):
    _, marginal, pi = specification(scenario)
    states = np.array(list(product(range(3), range(3), range(2))), int)
    mass = np.prod(np.stack([marginal[j][states[:, j]] for j in range(3)]), axis=0)
    z1 = np.array([-1., 0., 1.])[states[:, 0]]
    z2 = np.array([-1., 0., 1.])[states[:, 1]]
    z3 = np.array([-1., 1.])[states[:, 2]]
    if scenario == "additive":
        effects = np.c_[.9*z1 + .6*z2 + .35*z3, 1.2*z1 - .4*z2 + .65*z3]
    elif scenario == "rare_interaction":
        effects = np.c_[.65*z1 + .6*z2 + .6*z1*z3,
                         .45*z1 + .9*z2 - .75*z1*z3]
    else:
        effects = np.zeros((len(states), 2))

    def probabilities(intercepts):
        return softmax(np.c_[np.zeros(len(states)), effects + intercepts], axis=1)

    fit = root(lambda intercepts: (mass @ probabilities(intercepts) - pi)[1:], np.log(pi[1:]/pi[0]))
    if not fit.success:
        raise RuntimeError(f"Population intercept calibration failed: {fit.message}")
    probability = probabilities(fit.x)
    np.testing.assert_allclose(mass @ probability, pi, atol=1e-11, rtol=0)
    return states, mass, probability, pi


def generate(scenario, n=1500):
    levels, marginal, _ = specification(scenario)
    rng = np.random.default_rng(adapter.seed(MASTER, scenario, "sample"))
    ids = np.column_stack([rng.choice(len(v), n, p=m) for v, m in zip(levels, marginal)])
    frame = pd.DataFrame({f: np.asarray(v)[ids[:, j]] for j, (f, v) in enumerate(zip(FEATURES, levels))})
    states, _, probability, _ = population(scenario)
    lookup = {tuple(v): i for i, v in enumerate(states)}
    p = probability[[lookup[tuple(v)] for v in ids[:, :3]]]
    y = (rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1).astype(int)
    if np.bincount(y, minlength=3).min() < 10:
        raise ValueError("Insufficient class coverage; do not redraw this seed")
    core = adapter.get_core()
    frame[core.STABLE_RECORD_ID_COL] = np.arange(n, dtype=np.int64)
    schema = core.FeatureSchema(FEATURES.copy(), NUMERIC.copy(), [f for f in FEATURES if f not in NUMERIC])
    return frame, y, schema


def population_scores(scenario, fit_prior):
    """Exact raw-source targets, including the balanced-training probability map."""
    states, mass, p, pi = population(scenario)
    fit_prior = np.asarray(fit_prior, float)
    if fit_prior.shape != (3,) or (fit_prior <= 0).any():
        raise ValueError("Invalid fitting priors")

    def adjusted(prob):
        value = prob / fit_prior
        return value / value.sum(axis=-1, keepdims=True)

    q = adjusted(p)
    positive = .5 * mass[:, None] * p / pi
    negative = .5 * mass[:, None] * (1-p) / (1-pi)
    weight = positive + negative

    def loss(pred):
        return np.sum(positive*(1-pred)**2 + negative*pred**2, axis=0)

    lookup = {tuple(v): i for i, v in enumerate(states)}
    phi = np.zeros((len(states), 3, 3))
    for i, state in enumerate(states):
        game = {}
        for bits in product((False, True), repeat=3):
            hybrid = np.where(bits, state, states)
            game[bits] = mass @ q[[lookup[tuple(v)] for v in hybrid]]
        for j in range(3):
            for subset, value in game.items():
                if subset[j]:
                    continue
                added = list(subset)
                added[j] = True
                size = sum(subset)
                phi[i, j] += factorial(size)*factorial(2-size)/6 * (game[tuple(added)]-value)
    np.testing.assert_allclose(phi.sum(axis=1), q-mass@q, atol=1e-13, rtol=0)
    scores = np.zeros((4, 3, 10))
    scores[0, :, :3] = np.einsum("ic,ijc->cj", weight, abs(phi))
    base = loss(q)
    for j in range(3):
        pfi, means, weights = np.zeros(3), [], []
        for value in np.unique(states[:, j]):
            selected = states[:, j] == value
            hybrid = states.copy()
            hybrid[:, j] = value
            pfi += mass[selected].sum()*loss(q[[lookup[tuple(v)] for v in hybrid]])
            w = weight[selected].sum(axis=0)
            weights.append(w)
            means.append((weight[selected]*q[selected]).sum(axis=0)/w)
        scores[1, :, j] = pfi-base
        others = [k for k in range(3) if k != j]
        reduced = []
        for state in states:
            selected = (states[:, others] == state[others]).all(axis=1)
            reduced.append(adjusted(mass[selected] @ p[selected] / mass[selected].sum()))
        scores[2, :, j] = loss(np.array(reduced))-base
        means, weights = np.asarray(means), np.asarray(weights)
        scores[3, :, j] = np.sqrt((weights*(means-(weights*means).sum(axis=0))**2).sum(axis=0))
    return scores


def truth(scores, own=None):
    selected = scores if own is None else scores[[own]]
    return ((selected[..., :, None]-selected[..., None, :]) > 1e-10).all(axis=0)


def grade(graph, scores, own=None):
    delta = (scores if own is None else scores[[own]])
    delta = delta[..., :, None]-delta[..., None, :]
    correct = (delta > 1e-10).all(axis=0)
    ties = (abs(delta) <= 1e-10).all(axis=0)
    reverse = (delta < -1e-10).all(axis=0)
    false = graph & ~correct
    return dict(correct=int((graph & correct).sum()), wrong=int(false.sum()),
                declared=int(graph.sum()), true_total=int(correct.sum()),
                tied_wrong=int((false & ties).sum()), reverse_wrong=int((false & reverse).sum()),
                disagreement_wrong=int((false & ~(ties | reverse)).sum()))

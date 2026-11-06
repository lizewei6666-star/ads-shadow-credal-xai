"""Fitting-set shadow controls, paired orientations, and resampling roles."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MODELS = ("lr", "rf", "et", "hgb", "xgb", "lgbm", "catboost")
SOURCES = ("SHAP", "PFI", "LOCO", "FIRM")
EPS = 1e-12


def get_core():
    from . import numerics

    return numerics


def seed(*parts):
    # Seed labels are fixed so module renaming does not change random draws.
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16) % (2**31 - 1)


def partitions(y):
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    return [np.asarray(test, int) for _, test in splitter.split(np.zeros(len(y)), y)]


def split_roles(folds, y, partition, repeat=0):
    if sorted(np.concatenate(folds).tolist()) != list(range(len(y))):
        raise ValueError("Folds do not partition records")
    if not 0 <= partition < len(folds) or repeat < 0:
        raise ValueError("Invalid partition or repeat")
    draws = []
    for f, ids in enumerate(folds):
        ids = np.asarray(ids, int)
        if repeat == 0:
            draw = ids.copy()
        else:
            blocks = []
            for c in range(3):
                selected = ids[y[ids] == c]
                if not len(selected):
                    raise ValueError("Empty class stratum")
                rng = np.random.default_rng(seed("ads_bootstrap_v1", repeat, f, c))
                blocks.append(rng.choice(selected, len(selected), replace=True))
            draw = np.concatenate(blocks)
            np.random.default_rng(seed("ads_bootstrap_shuffle_v1", repeat, f)).shuffle(draw)
        np.testing.assert_array_equal(
            np.bincount(y[draw], minlength=3), np.bincount(y[ids], minlength=3)
        )
        draws.append(draw)
    cal = (partition + 1) % len(folds)
    fit = np.concatenate([draw for f, draw in enumerate(draws) if f not in (partition, cal)])
    result = (fit, draws[cal], draws[partition])
    if any(set(a) & set(b) for a, b in combinations(result, 2)):
        raise ValueError("Record identities overlap across roles")
    return result


def control_names(features):
    return {f: (f"ShadowA__{f}", f"ShadowB__{f}") for f in features}


def augment(raw, schema, fit_ids, partition, repeat=0):
    core = get_core()
    features = list(schema.feature_cols)
    names = control_names(features)
    controls = [name for f in features for name in names[f]]
    if set(controls) & set(raw.columns):
        raise ValueError("Control names collide with original columns")
    if not len(fit_ids):
        raise ValueError("Empty fitting set")
    result = raw.copy(deep=True)
    provenance = {}
    for feature in features:
        if raw.iloc[fit_ids][feature].isna().any():
            raise ValueError(f"Missing fitting values in {feature}")
        for name in names[feature]:
            s = seed("ads_matched_controls_v1", partition, repeat, name)
            donor = np.random.default_rng(s).choice(fit_ids, len(raw), replace=True)
            result[name] = raw.iloc[donor][feature].to_numpy(copy=True)
            provenance[name] = dict(parent=feature, seed=s, donor_ids=donor)
    numerical = list(schema.numeric_cols) + [
        z for f in features if f in schema.numeric_cols for z in names[f]
    ]
    categorical = list(schema.categorical_cols) + [
        z for f in features if f in schema.categorical_cols for z in names[f]
    ]
    new_schema = core.FeatureSchema(features + controls, numerical, categorical)
    pd.testing.assert_frame_equal(result[raw.columns], raw)
    return result, new_schema, provenance


def oriented(raw, features, orientation):
    if orientation not in (0, 1):
        raise ValueError("Orientation must be zero or one")
    result = raw.copy(deep=True)
    if orientation:
        for a, b in control_names(features).values():
            result[[a, b]] = raw[[b, a]].to_numpy()
    return result


def aligned_names(features, schema, orientation):
    swap = {}
    if orientation:
        for a, b in control_names(features).values():
            swap.update({a: b, b: a})
    return [swap.get(f, f) for f in schema.feature_cols]


def control_positions(features, columns):
    return np.array(
        [[columns.index(name) for name in pair] for pair in control_names(features).values()],
        dtype=int,
    )


def votes(center, p, controls, gate="shadow"):
    center = np.asarray(center, float)
    controls = np.asarray(controls, int)
    if center.ndim != 5 or center.shape[1:4] != (2, 4, 3) or not np.isfinite(center).all():
        raise ValueError("Expected finite repeat,block,source,class,coordinate scores")
    if not 1 <= p <= center.shape[-1] or controls.shape != (p, 2):
        raise ValueError("Invalid feature/control dimensions")
    if gate not in ("none", "zero", "shadow"):
        raise ValueError("Unknown gate")
    scores = center[..., :p]
    direction = scores[..., :, None] - scores[..., None, :] > EPS
    if gate == "shadow":
        if (controls < p).any() or (controls >= center.shape[-1]).any():
            raise ValueError("Controls must reference non-scientific coordinates")
        qualified = (scores[..., :, None] - center[..., controls] > EPS).all(axis=-1)
    elif gate == "zero":
        qualified = scores > EPS
    else:
        qualified = np.ones_like(scores, dtype=bool)
    return (direction & qualified[..., :, None]).all(axis=1)


def joint_graphs(source_votes):
    support = source_votes.sum(axis=1)
    passed = (support >= 2) & (support > support.swapaxes(-1, -2))
    anchor = passed[0]
    reach = anchor.copy()
    for j in range(anchor.shape[-1]):
        reach |= reach[..., :, j, None] & reach[..., None, j, :]
    mask = ~(reach & reach.swapaxes(-1, -2))
    counts = passed[1:].sum(axis=0)
    graphs = np.array([anchor & mask & (counts >= q) for q in range(len(passed))])
    return dict(graphs=graphs, counts=counts, anchor=anchor, mask=mask)


@contextmanager
def preflight_core():
    core = get_core()
    changes = dict(
        MODEL_N_JOBS=1,
        SHAP_BACKGROUND_SIZE=12,
        SHAP_PERMUTATION_CYCLES=1,
        PERMUTATION_REPEATS=3,
        TARGET_SHAP_PREDICTION_BATCH_ROWS=4096,
        ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT=False,
        ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT=False,
    )
    previous = {name: getattr(core, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(core, name, value)
        yield core
    finally:
        for name, value in previous.items():
            setattr(core, name, value)

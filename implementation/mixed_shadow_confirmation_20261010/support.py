"""Fresh samples and isolated reuse of the frozen mixed-data implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
DISCOVERY = WORK / "mixed_shadow_resampling_20261010"
OUT = ROOT / "outputs"
POINT = OUT / "original"
MASTER = 2026101002
PRIMARY = 5
SENSITIVITY = (4, 6, 10, 11, 12)
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "ads-shadow-credal-xai"))


def load_module(name, path):
    path = Path(path)
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


previous = load_module("frozen_mixed_resampling", DISCOVERY / "run.py")
base = previous.base
design = sys.modules[base.generate.__module__]
adapter = base.adapter
SCENARIOS, SOURCES, FEATURES = base.SCENARIOS, base.SOURCES, base.FEATURES
sha, digest, read, write, immutable = base.sha, base.digest, base.read, base.write, base.immutable
recurrence, ratios = previous.recurrence, previous.ratios

import numpy as np
import pandas as pd


def sample(scenario, master=MASTER, n=1500):
    """The frozen generator with the sample master seed made explicit."""
    levels, marginal, _ = design.specification(scenario)
    rng = np.random.default_rng(adapter.seed(master, scenario, "sample"))
    ids = np.column_stack([rng.choice(len(v), n, p=m) for v, m in zip(levels, marginal)])
    frame = pd.DataFrame({f: np.asarray(v)[ids[:, j]] for j, (f, v) in enumerate(zip(FEATURES, levels))})
    states, _, probability, _ = design.population(scenario)
    lookup = {tuple(v): i for i, v in enumerate(states)}
    p = probability[[lookup[tuple(v)] for v in ids[:, :3]]]
    y = (rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1).astype(int)
    if np.bincount(y, minlength=3).min() < 10:
        raise ValueError("Insufficient class coverage; do not redraw this seed")
    core = adapter.get_core()
    frame[core.STABLE_RECORD_ID_COL] = np.arange(n, dtype=np.int64)
    schema = core.FeatureSchema(FEATURES.copy(), design.NUMERIC.copy(), [f for f in FEATURES if f not in design.NUMERIC])
    return frame, y, schema


def bind(function, **dependencies):
    """Reuse estimator bytecode without mutating the archived module globals."""
    namespace = dict(function.__globals__, **dependencies)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def bindings(verifier, status, out=OUT, point=POINT, generator=sample):
    namespace = SimpleNamespace(**dict(vars(base), OUT=point, generate=generator))
    roles = bind(previous.roles_for, base=namespace)
    checkpoint = bind(previous.load_unit, OUT=out)
    return SimpleNamespace(
        prepare=bind(base.prepare, OUT=point, generate=generator, status=status),
        point_worker=bind(base.worker, OUT=point, generate=generator, verify_contract=verifier),
        repeat_worker=bind(previous.worker, OUT=out, base=namespace,
                           verify_contract=verifier, roles_for=roles, load_unit=checkpoint),
        roles=roles,
        checkpoint=checkpoint,
    )


def csv(path):
    return pd.read_csv(path, keep_default_na=False,
                       na_values={"precision": [""], "recovery": [""]})

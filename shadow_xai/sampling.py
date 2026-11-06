"""Simulation sampling and per-run worker bindings."""

from types import FunctionType, SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from . import controls as adapter
from . import population as design
from . import resampling as previous
from . import simulation_point as base
from .population import FEATURES, SCENARIOS

MASTER = 2026101002
OUT = Path("outputs/simulation")
POINT = OUT / "admission"
SOURCES = adapter.SOURCES


def sample(scenario, master=MASTER, n=1500):
    """Draw one dataset using the specified master seed and population model."""
    levels, marginal, _ = design.specification(scenario)
    rng = np.random.default_rng(adapter.seed(master, scenario, "sample"))
    ids = np.column_stack([rng.choice(len(v), n, p=m) for v, m in zip(levels, marginal)])
    frame = pd.DataFrame(
        {f: np.asarray(v)[ids[:, j]] for j, (f, v) in enumerate(zip(FEATURES, levels))}
    )
    states, _, probability, _ = design.population(scenario)
    lookup = {tuple(v): i for i, v in enumerate(states)}
    p = probability[[lookup[tuple(v)] for v in ids[:, :3]]]
    y = (rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1).astype(int)
    if np.bincount(y, minlength=3).min() < 10:
        raise ValueError("Insufficient class coverage; do not redraw this seed")
    core = adapter.get_core()
    frame[core.STABLE_RECORD_ID_COL] = np.arange(n, dtype=np.int64)
    schema = core.FeatureSchema(
        FEATURES.copy(), design.NUMERIC.copy(), [f for f in FEATURES if f not in design.NUMERIC]
    )
    return frame, y, schema


def bind(function, **dependencies):
    """Bind run-specific dependencies without changing shared module globals."""
    namespace = dict(function.__globals__, **dependencies)
    result = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def bindings(verifier, status, out=OUT, point=POINT, generator=sample):
    namespace = SimpleNamespace(**dict(vars(base), OUT=point, generate=generator))
    roles = bind(previous.roles_for, base=namespace)
    checkpoint = bind(previous.load_unit, OUT=out)
    return SimpleNamespace(
        prepare=bind(base.prepare, OUT=point, generate=generator, status=status),
        point_worker=bind(base.worker, OUT=point, generate=generator, verify_contract=verifier),
        repeat_worker=bind(
            previous.worker,
            OUT=out,
            base=namespace,
            verify_contract=verifier,
            roles_for=roles,
            load_unit=checkpoint,
        ),
        roles=roles,
        checkpoint=checkpoint,
    )

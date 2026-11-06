"""Model admission and original-sample source profiles for simulation."""

from __future__ import annotations
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import controls as adapter
from .population import FEATURES, SCENARIOS
from .selection_cache import cached_selection

SOURCES = adapter.SOURCES
OUT = Path("outputs/simulation/admission")


def generate(scenario):
    raise RuntimeError("Use the configured simulation runner")


def verify_contract():
    raise RuntimeError("Use the configured simulation runner")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen file changed: {path}")
    else:
        write(path, value)


def status(phase, **detail):
    record = dict(phase=phase, **detail)
    write(OUT / "progress.json", record)
    print(json.dumps(record), flush=True)


@contextmanager
def settings():
    with adapter.preflight_core() as core:
        changes = dict(
            CANDIDATE_MODELS=list(adapter.MODELS),
            INNER_CV_FOLDS=5,
            OUTPUT_FIGURES=False,
            ENABLE_CALIBRATION_DIAGNOSTICS=False,
        )
        previous = {k: getattr(core, k) for k in changes}
        try:
            for k, value in changes.items():
                setattr(core, k, value)
            yield core
        finally:
            for k, value in previous.items():
                setattr(core, k, value)


def prepare(scenario, identity):
    folder = OUT / scenario
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    saved = folder / "prepared.json"
    if saved.exists():
        value = read(saved)
        assert value["identity"] == identity
        for name, checksum in value["files"].items():
            assert sha(folder / name) == checksum
        return value
    raw, y, schema = generate(scenario)
    folds = adapter.partitions(y)
    fit, cal, ext = adapter.split_roles(folds, y, 0)
    selection_ids = np.concatenate([ids for i, ids in enumerate(folds) if i != 0])
    assert np.bincount(y[selection_ids], minlength=3).min() >= 5
    part = adapter.get_core().OuterPartition(1, 1, 1, 42, selection_ids, ext)
    raw.assign(outcome=y).to_csv(folder / "synthetic.csv", index=False)
    write(
        folder / "roles.json",
        dict(fit=fit.tolist(), calibration=cal.tolist(), external=ext.tolist()),
    )
    status("admission", scenario=scenario, classes=np.bincount(y, minlength=3).tolist())
    started = time.perf_counter()
    with (
        settings() as core,
        threadpool_limits(limits=1),
        (folder / "admission.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        selection, _, _ = cached_selection(
            core, raw, y, schema, [part], folder, digest([identity, scenario])
        )
        assert set(selection.model) == set(adapter.MODELS)
        assert selection.n_folds.eq(5).all()
        admitted = core.select_explanation_models_by_seed(selection, [1])[1]
    record = dict(
        identity=identity,
        scenario=scenario,
        admitted=admitted,
        classes=np.bincount(y, minlength=3).tolist(),
        role_classes={
            name: np.bincount(y[ids], minlength=3).tolist()
            for name, ids in zip(("fit", "calibration", "external"), (fit, cal, ext))
        },
        seconds=time.perf_counter() - started,
        files={
            name: sha(folder / name)
            for name in ("synthetic.csv", "roles.json", "selection_cache.json")
        },
    )
    immutable(saved, record)
    status("admitted", scenario=scenario, models=admitted, seconds=round(record["seconds"], 2))
    return record


def worker(task):
    scenario, model, arm, identity = task
    verify_contract()
    folder = OUT / scenario
    key = f"{arm}_{model}"
    record_path = folder / f"{key}.json"
    if record_path.exists():
        value = read(record_path)
        assert value["identity"] == identity
        for name, checksum in value["files"].items():
            assert sha(folder / name) == checksum
        return dict(scenario=scenario, key=key, cached=True, seconds=value["seconds"])
    raw, y, schema = generate(scenario)
    prepared = read(folder / "prepared.json")
    assert prepared["identity"] == identity and model in prepared["admitted"]
    sample = raw.assign(outcome=y).to_csv(index=False)
    # Compare parsed values, including stable record IDs, with the frozen sample.
    stored = pd.read_csv(folder / "synthetic.csv")
    pd.testing.assert_frame_equal(pd.read_csv(__import__("io").StringIO(sample)), stored)
    roles = read(folder / "roles.json")
    fit, cal, ext = [np.asarray(roles[k], int) for k in ("fit", "calibration", "external")]
    features = schema.feature_cols.copy()
    donors = {}
    if arm == "augmented":
        raw, schema, donors = adapter.augment(raw, schema, fit, 0)
        assert all(set(d["donor_ids"]) <= set(fit) for d in donors.values())
    elif arm != "plain":
        raise ValueError(arm)
    started = time.perf_counter()
    profiles, rows = [], []
    with (
        (folder / f"{key}.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
        settings() as core,
        threadpool_limits(limits=1),
    ):
        for orientation in range(2 if arm == "augmented" else 1):
            work = adapter.oriented(raw, features, orientation) if arm == "augmented" else raw
            train = work.iloc[fit].reset_index(drop=True)
            evaluation = work.iloc[np.r_[cal, ext]].reset_index(drop=True)
            fit_seed = core.stable_seed_from_key("fixed_spec_outer_fit", 1, model)
            source_seed = core.stable_seed_from_key("fixed_source_protocol_outer", 1, model)
            estimator, encoded, probability, _ = core.fit_fixed_predictive_specification(
                train, evaluation, y[fit], schema, model, fit_seed, [0, 1, 2]
            )
            codec = core.RawFeatureCodec.fit(train, schema)
            np.testing.assert_array_equal(
                core.transform_raw(encoded.preprocessor, codec.decode(codec.encode(evaluation))),
                encoded.X_test,
            )
            reduced = {}
            for feature in schema.feature_cols:
                print(
                    json.dumps(dict(phase="loco", orientation=orientation, feature=feature)),
                    flush=True,
                )
                reduced[feature] = core.fit_loco_reduced_probability(
                    y_train=y[fit],
                    X_train_raw=train,
                    X_eval_raw=evaluation,
                    schema=schema,
                    classes=[0, 1, 2],
                    model_name=model,
                    feature=feature,
                    seed=source_seed,
                    n_jobs=1,
                    sample_weight=core.make_balanced_sample_weight(y[fit]),
                )
            blocks = []
            part = core.OuterPartition(1, 1, 1, 42, fit, ext)
            for block, segment in (
                ("calibration", slice(0, len(cal))),
                ("external", slice(len(cal), None)),
            ):
                ev = evaluation.iloc[segment].reset_index(drop=True)
                tables, errors = core.compute_fitted_source_tables(
                    estimator=estimator,
                    encoded=replace(encoded, X_test=encoded.X_test[segment]),
                    proba=probability[segment],
                    y_train=y[fit],
                    y_eval=y[np.r_[cal, ext]][segment],
                    X_train_raw=train,
                    X_eval_raw=ev,
                    schema=schema,
                    class_order=[0, 1, 2],
                    model_name=model,
                    source_seed=source_seed,
                    feature_cols=schema.feature_cols,
                    worker_root=folder,
                    call_key=f"{key}_{orientation}_{block}",
                    include_firm_table=True,
                    shap_phase_scope="outer_actual",
                    outer_partition=1,
                    inner_fold=-1,
                    loco_reduced_proba_by_feature={f: a[segment] for f, a in reduced.items()},
                )
                if errors or set(tables) != set(SOURCES):
                    raise RuntimeError(f"Missing explanation source: {errors}")
                table = pd.concat(
                    [
                        core.standardize_outer_source_table(s, tables[s], part, model)
                        for s in SOURCES
                    ],
                    ignore_index=True,
                )
                assert len(table) == 12 * len(schema.feature_cols)
                assert not table.duplicated(["source", "class_id", "feature"]).any()
                assert np.isfinite(table.raw_statistic).all()
                assert all(t.n_eval.eq(len(ev)).all() for t in tables.values())
                keyed = table.set_index(["source", "class_id", "feature"]).raw_statistic
                order = (
                    adapter.aligned_names(features, schema, orientation)
                    if arm == "augmented"
                    else schema.feature_cols
                )
                blocks.append(
                    np.array(
                        [keyed[s, c, f] for s in SOURCES for c in range(3) for f in order]
                    ).reshape(4, 3, -1)
                )
                rows.append(
                    table[["source", "class_id", "feature", "raw_statistic"]].assign(
                        block=block, orientation=orientation
                    )
                )
            profiles.append(blocks)
            del estimator, encoded, reduced
    profile = np.asarray(profiles)
    assert profile.shape == ((2, 2, 4, 3, 30) if arm == "augmented" else (1, 2, 4, 3, 10))
    np.savez_compressed(folder / f"{key}.npz", orientations=profile, mean=profile.mean(axis=0))
    pd.concat(rows).to_csv(folder / f"{key}.csv", index=False)
    control_audit = {
        f: dict(
            parent=d["parent"],
            donor_hash=digest(d["donor_ids"].tolist()),
            fit_levels=int(raw.iloc[fit][f].nunique()),
            parent_fit_levels=int(raw.iloc[fit][d["parent"]].nunique()),
        )
        for f, d in donors.items()
    }
    result = dict(
        identity=identity,
        scenario=scenario,
        model=model,
        arm=arm,
        seconds=time.perf_counter() - started,
        source_rows=sum(len(t) for t in rows),
        controls=control_audit,
        files={f"{key}{ext}": sha(folder / f"{key}{ext}") for ext in (".npz", ".csv")},
    )
    immutable(record_path, result)
    return dict(scenario=scenario, key=key, cached=False, seconds=result["seconds"])

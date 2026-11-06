"""Bootstrap source profiles with the original model admission held fixed."""

from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import simulation_point as base
from .simulation_point import sha, digest, read, write, immutable

adapter = base.adapter
SOURCES = adapter.SOURCES
OUT = Path("outputs/simulation")


def verify_contract():
    raise RuntimeError("Use the configured simulation runner")


def key_for(task):
    scenario, model, arm, repeat, _ = task
    return f"{scenario}__{arm}__{model}__r{repeat:02d}"


def load_unit(task):
    key = key_for(task)
    folder = OUT / "units" / key
    path = folder / "complete.json"
    if not path.exists():
        return None
    value = read(path)
    assert value["identity"] == task[-1] and value["key"] == key
    for name, checksum in value["files"].items():
        assert sha(folder / name) == checksum, (key, name)
    return value


def roles_for(scenario, repeat):
    raw, y, schema = base.generate(scenario)
    roles = adapter.split_roles(adapter.partitions(y), y, 0, repeat)
    original = read(base.OUT / scenario / "roles.json")
    for ids, name in zip(roles, ("fit", "calibration", "external")):
        original_ids = np.asarray(original[name], int)
        assert set(ids) <= set(original_ids)
        np.testing.assert_array_equal(
            np.bincount(y[ids], minlength=3), np.bincount(y[original_ids], minlength=3)
        )
        if repeat == 0:
            np.testing.assert_array_equal(ids, original_ids)
    return raw, y, schema, roles


def worker(task):
    scenario, model, arm, repeat, identity = task
    assert verify_contract() == identity
    key = key_for(task)
    saved = load_unit(task)
    if saved:
        return dict(key=key, cached=True, seconds=saved["seconds"])
    folder = OUT / "units" / key
    folder.mkdir(parents=True, exist_ok=True)
    raw, y, schema, (fit, cal, ext) = roles_for(scenario, repeat)
    prepared = read(base.OUT / scenario / "prepared.json")
    assert model in prepared["admitted"]
    features = schema.feature_cols.copy()
    donors = {}
    if arm == "augmented":
        raw, schema, donors = adapter.augment(raw, schema, fit, 0, repeat)
        assert all(set(d["donor_ids"]) <= set(fit) for d in donors.values())
    elif arm != "plain":
        raise ValueError(arm)
    roles = dict(zip(("fit", "calibration", "external"), [v.tolist() for v in (fit, cal, ext)]))
    immutable(folder / "roles.json", roles)
    started = time.perf_counter()
    profiles, rows = [], []
    with (
        (folder / "execution.log").open("w", encoding="utf-8") as log,
        redirect_stdout(log),
        redirect_stderr(log),
        base.settings() as core,
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
                    base.json.dumps(dict(phase="loco", orientation=orientation, feature=feature)),
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
                    call_key=f"{arm}_{model}_{orientation}_{block}",
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
    if repeat == 0:
        with np.load(base.OUT / scenario / f"{arm}_{model}.npz", allow_pickle=False) as parent:
            np.testing.assert_array_equal(profile, parent["orientations"])
        original_rows = pd.read_csv(base.OUT / scenario / f"{arm}_{model}.csv")
        pd.testing.assert_frame_equal(
            pd.concat(rows, ignore_index=True),
            original_rows,
            check_exact=False,
            rtol=1e-13,
            atol=1e-14,
        )
    np.savez_compressed(folder / "profile.npz", orientations=profile, mean=profile.mean(axis=0))
    pd.concat(rows).to_csv(folder / "sources.csv", index=False)
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
        key=key,
        scenario=scenario,
        model=model,
        arm=arm,
        repeat=repeat,
        seconds=time.perf_counter() - started,
        source_rows=sum(len(t) for t in rows),
        controls=control_audit,
        files={name: sha(folder / name) for name in ("profile.npz", "sources.csv", "roles.json")},
    )
    immutable(folder / "complete.json", result)
    return dict(key=key, cached=False, seconds=result["seconds"])

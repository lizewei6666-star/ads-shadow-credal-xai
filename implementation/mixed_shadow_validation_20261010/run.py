"""Resumable paired mixed-data pilot; population grading follows frozen graphs."""
from __future__ import annotations

from design import WORK, SCENARIOS, MASTER, FEATURES, generate, population_scores, grade, truth, adapter
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from validation.execution import cached_selection

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SOURCES = adapter.SOURCES


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
        changes = dict(CANDIDATE_MODELS=list(adapter.MODELS), INNER_CV_FOLDS=5,
                       OUTPUT_FIGURES=False, ENABLE_CALIBRATION_DIAGNOSTICS=False)
        previous = {k: getattr(core, k) for k in changes}
        try:
            for k, value in changes.items():
                setattr(core, k, value)
            yield core
        finally:
            for k, value in previous.items():
                setattr(core, k, value)


def contract():
    core = adapter.get_core()
    from validation import core_adapter, execution
    files = [ROOT / name for name in ("PROTOCOL.md", "design.py", "run.py")]
    files += [Path(m.__file__) for m in (adapter, core, core_adapter, execution)]
    body = dict(version="mixed_matched_pilot_v1", seed=MASTER, n=1500, p=10,
                scenarios=list(SCENARIOS), outer_partitions=[1], record_repeats=[0],
                models=list(adapter.MODELS), background=12, shap_cycles=1, pfi_repeats=3,
                files={str(p): sha(p) for p in files},
                versions={p: importlib.metadata.version(p) for p in
                          ("numpy", "pandas", "scipy", "scikit-learn", "shap", "xgboost", "lightgbm", "catboost")})
    value = dict(identity=digest(body), contract=body)
    immutable(OUT / "contract.json", value)
    return value["identity"]


def verify_contract():
    value = read(OUT / "contract.json")
    assert digest(value["contract"]) == value["identity"]
    for name, checksum in value["contract"]["files"].items():
        assert sha(name) == checksum, name
    return value["identity"]


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
    write(folder / "roles.json", dict(fit=fit.tolist(), calibration=cal.tolist(), external=ext.tolist()))
    status("admission", scenario=scenario, classes=np.bincount(y, minlength=3).tolist())
    started = time.perf_counter()
    with settings() as core, threadpool_limits(limits=1), (folder / "admission.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        selection, _, _ = cached_selection(core, raw, y, schema, [part], folder, digest([identity, scenario]))
        assert set(selection.model) == set(adapter.MODELS)
        assert selection.n_folds.eq(5).all()
        admitted = core.select_explanation_models_by_seed(selection, [1])[1]
    record = dict(identity=identity, scenario=scenario, admitted=admitted,
                  classes=np.bincount(y, minlength=3).tolist(),
                  role_classes={name: np.bincount(y[ids], minlength=3).tolist()
                                for name, ids in zip(("fit", "calibration", "external"), (fit, cal, ext))},
                  seconds=time.perf_counter()-started,
                  files={name: sha(folder / name) for name in ("synthetic.csv", "roles.json", "selection_cache.json")})
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
    with (folder / f"{key}.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log), settings() as core, threadpool_limits(limits=1):
        for orientation in range(2 if arm == "augmented" else 1):
            work = adapter.oriented(raw, features, orientation) if arm == "augmented" else raw
            train = work.iloc[fit].reset_index(drop=True)
            evaluation = work.iloc[np.r_[cal, ext]].reset_index(drop=True)
            fit_seed = core.stable_seed_from_key("fixed_spec_outer_fit", 1, model)
            source_seed = core.stable_seed_from_key("fixed_source_protocol_outer", 1, model)
            estimator, encoded, probability, _ = core.fit_fixed_predictive_specification(
                train, evaluation, y[fit], schema, model, fit_seed, [0, 1, 2])
            codec = core.RawFeatureCodec.fit(train, schema)
            np.testing.assert_array_equal(core.transform_raw(encoded.preprocessor, codec.decode(codec.encode(evaluation))), encoded.X_test)
            reduced = {}
            for feature in schema.feature_cols:
                print(json.dumps(dict(phase="loco", orientation=orientation, feature=feature)), flush=True)
                reduced[feature] = core.fit_loco_reduced_probability(
                    y_train=y[fit], X_train_raw=train, X_eval_raw=evaluation, schema=schema,
                    classes=[0, 1, 2], model_name=model, feature=feature, seed=source_seed,
                    n_jobs=1, sample_weight=core.make_balanced_sample_weight(y[fit]))
            blocks = []
            part = core.OuterPartition(1, 1, 1, 42, fit, ext)
            for block, segment in (("calibration", slice(0, len(cal))), ("external", slice(len(cal), None))):
                ev = evaluation.iloc[segment].reset_index(drop=True)
                tables, errors = core.compute_fitted_source_tables(
                    estimator=estimator, encoded=replace(encoded, X_test=encoded.X_test[segment]),
                    proba=probability[segment], y_train=y[fit], y_eval=y[np.r_[cal, ext]][segment],
                    X_train_raw=train, X_eval_raw=ev, schema=schema, class_order=[0, 1, 2],
                    model_name=model, source_seed=source_seed, feature_cols=schema.feature_cols,
                    worker_root=folder, call_key=f"{key}_{orientation}_{block}",
                    include_firm_table=True, shap_phase_scope="outer_actual", outer_partition=1,
                    inner_fold=-1, loco_reduced_proba_by_feature={f: a[segment] for f, a in reduced.items()})
                if errors or set(tables) != set(SOURCES):
                    raise RuntimeError(f"Missing explanation source: {errors}")
                table = pd.concat([core.standardize_outer_source_table(s, tables[s], part, model) for s in SOURCES], ignore_index=True)
                assert len(table) == 12*len(schema.feature_cols)
                assert not table.duplicated(["source", "class_id", "feature"]).any()
                assert np.isfinite(table.raw_statistic).all()
                assert all(t.n_eval.eq(len(ev)).all() for t in tables.values())
                keyed = table.set_index(["source", "class_id", "feature"]).raw_statistic
                order = adapter.aligned_names(features, schema, orientation) if arm == "augmented" else schema.feature_cols
                blocks.append(np.array([keyed[s, c, f] for s in SOURCES for c in range(3) for f in order]).reshape(4, 3, -1))
                rows.append(table[["source", "class_id", "feature", "raw_statistic"]].assign(block=block, orientation=orientation))
            profiles.append(blocks)
            del estimator, encoded, reduced
    profile = np.asarray(profiles)
    assert profile.shape == ((2, 2, 4, 3, 30) if arm == "augmented" else (1, 2, 4, 3, 10))
    np.savez_compressed(folder / f"{key}.npz", orientations=profile, mean=profile.mean(axis=0))
    pd.concat(rows).to_csv(folder / f"{key}.csv", index=False)
    control_audit = {f: dict(parent=d["parent"], donor_hash=digest(d["donor_ids"].tolist()),
                            fit_levels=int(raw.iloc[fit][f].nunique()),
                            parent_fit_levels=int(raw.iloc[fit][d["parent"]].nunique())) for f, d in donors.items()}
    result = dict(identity=identity, scenario=scenario, model=model, arm=arm,
                  seconds=time.perf_counter()-started, source_rows=sum(len(t) for t in rows),
                  controls=control_audit, files={f"{key}{ext}": sha(folder / f"{key}{ext}") for ext in (".npz", ".csv")})
    immutable(record_path, result)
    return dict(scenario=scenario, key=key, cached=False, seconds=result["seconds"])


def graphs(center, arm, gate, method):
    controls = adapter.control_positions(FEATURES, FEATURES + [n for pair in adapter.control_names(FEATURES).values() for n in pair])
    padded = np.pad(center, [(0, 0)]*3 + [(0, 20)]) if arm == "plain" else center
    v = adapter.votes(padded[None], 10, controls, gate)
    if method == "Joint4":
        return adapter.joint_graphs(v)["graphs"][0]
    return v[0, SOURCES.index(method)]


def configurations():
    for arm in ("plain", "augmented"):
        for gate in (("none", "zero") if arm == "plain" else ("none", "zero", "shadow")):
            for method in (*SOURCES, "Joint4"):
                yield arm, gate, method


def evaluate(identity):
    if (OUT / "results_complete.json").exists():
        raise RuntimeError("Results already complete; run the independent audit")
    decisions, centers = {}, {}
    for scenario in SCENARIOS:
        folder = OUT / scenario
        admitted = read(folder / "prepared.json")["admitted"]
        for arm in ("plain", "augmented"):
            values = []
            for model in admitted:
                record = read(folder / f"{arm}_{model}.json")
                assert record["identity"] == identity
                for name, checksum in record["files"].items():
                    assert sha(folder / name) == checksum
                with np.load(folder / f"{arm}_{model}.npz", allow_pickle=False) as data:
                    values.append(data["mean"])
            centers[scenario, arm] = np.median(values, axis=0)
        for arm, gate, method in configurations():
            decisions[f"{scenario}__{arm}__{gate}__{method}"] = graphs(centers[scenario, arm], arm, gate, method)
    np.savez_compressed(OUT / "graphs.npz", **decisions)
    np.savez_compressed(OUT / "pooled_profiles.npz", **{f"{s}__{a}": v for (s, a), v in centers.items()})
    immutable(OUT / "empirical_frozen.json", dict(identity=identity,
              files={name: sha(OUT / name) for name in ("graphs.npz", "pooled_profiles.npz")}))
    refs, rows, paired = {}, [], []
    for scenario in SCENARIOS:
        raw, y, _ = generate(scenario)
        roles = read(OUT / scenario / "roles.json")
        fit_prior = np.bincount(y[roles["fit"]], minlength=3)/len(roles["fit"])
        refs[scenario] = population_scores(scenario, fit_prior)
        for arm, gate, method in configurations():
            g = decisions[f"{scenario}__{arm}__{gate}__{method}"]
            for target in (("common",) if method == "Joint4" else ("common", "own")):
                own = SOURCES.index(method) if target == "own" else None
                for c in range(3):
                    rows.append(dict(scenario=scenario, arm=arm, gate=gate, method=method, target=target,
                                     class_id=c, **grade(g[c], refs[scenario][:, c], own)))
            main = decisions[f"{scenario}__augmented__shadow__Joint4"]
            t = truth(refs[scenario])
            paired.append(dict(scenario=scenario, arm=arm, gate=gate, method=method,
                               shared_correct=int((main & g & t).sum()), shared_wrong=int((main & g & ~t).sum()),
                               main_only_correct=int((main & ~g & t).sum()), main_only_wrong=int((main & ~g & ~t).sum()),
                               comparator_only_correct=int((~main & g & t).sum()), comparator_only_wrong=int((~main & g & ~t).sum())))
    np.savez_compressed(OUT / "references.npz", **refs)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "classes.csv", index=False)
    counts = ["correct", "wrong", "declared", "true_total", "tied_wrong", "reverse_wrong", "disagreement_wrong"]
    for name, keys in (("by_scenario", ["scenario", "arm", "gate", "method", "target"]),
                       ("overall", ["arm", "gate", "method", "target"])):
        table = frame.groupby(keys)[counts].sum().reset_index()
        table["precision"] = table.correct/table.declared.replace(0, np.nan)
        table["recovery"] = table.correct/table.true_total.replace(0, np.nan)
        table.to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_comparisons.csv", index=False)
    verify_contract()
    write(OUT / "results_complete.json", dict(identity=identity, scope="one_partition_point_graph_diagnostic",
          scenarios=3, independent_datasets=3, graphs=len(decisions), rows=len(frame),
          files={p.name: sha(p) for p in OUT.iterdir() if p.suffix in (".csv", ".npz")}, status="complete"))
    status("evaluated", graphs=len(decisions), rows=len(frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "fit", "evaluate", "all"), default="all")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock = OUT / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    start = time.perf_counter()
    try:
        identity = contract()
        if args.stage in ("prepare", "all"):
            for scenario in SCENARIOS:
                prepare(scenario, identity)
        if args.stage in ("fit", "all"):
            tasks = [(s, m, a, identity) for s in SCENARIOS
                     for m in read(OUT / s / "prepared.json")["admitted"] for a in ("plain", "augmented")]
            costs = dict(lr=1, lgbm=2, xgb=3, catboost=4, et=5, rf=6, hgb=7)
            tasks.sort(key=lambda t: -costs[t[1]])
            status("fitting", total=len(tasks), jobs=args.jobs)
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(worker, t) for t in tasks]
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    status("unit_complete", done=i, total=len(tasks), **result)
            immutable(OUT / "fits_complete.json", dict(identity=identity, units=len(tasks), status="complete"))
        if args.stage in ("evaluate", "all"):
            assert read(OUT / "fits_complete.json")["identity"] == identity
            evaluate(identity)
        status("stage_complete", stage=args.stage, seconds=round(time.perf_counter()-start, 2))
    except BaseException as exc:
        write(OUT / f"failure_{int(time.time())}.json", dict(error=repr(exc), seconds=time.perf_counter()-start))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

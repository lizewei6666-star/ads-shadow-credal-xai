"""Frozen, resumable bootstrap extension of the paired mixed-data pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "mixed_shadow_validation_20261010"
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location("mixed_parent_runner", PARENT / "run.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from decisions import recurrence, ratios
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

adapter = base.adapter
SCENARIOS, SOURCES = base.SCENARIOS, base.SOURCES
OUT = ROOT / "outputs"
sha, digest, read, write, immutable = base.sha, base.digest, base.read, base.write, base.immutable
PARENT_ID = "0ae1f4eb5b1de6f98ec0d36b91cdb2e3d50f51fd01f282b290cd04b59f5a1610"


def contract():
    assert base.verify_contract() == PARENT_ID
    parent = base.OUT
    paths = [parent / "contract.json", parent / "independent_audit.json",
             parent / "results_complete.json", parent / "graphs.npz",
             parent / "references.npz", parent / "pooled_profiles.npz"]
    for scenario in SCENARIOS:
        folder = parent / scenario
        prepared = read(folder / "prepared.json")
        paths.append(folder / "prepared.json")
        paths.extend(folder / name for name in prepared["files"])
        for model in prepared["admitted"]:
            for arm in ("plain", "augmented"):
                key = f"{arm}_{model}"
                record = read(folder / f"{key}.json")
                paths.append(folder / f"{key}.json")
                paths.extend(folder / name for name in record["files"])
    paths.extend(ROOT / name for name in ("run.py", "decisions.py", "PROTOCOL.md", "test_resampling.py"))
    body = dict(version="mixed_matched_resampling_v1", parent_identity=PARENT_ID,
                repeats=list(range(13)), primary=10, sensitivity=[11, 12],
                files={str(p): sha(p) for p in paths},
                versions=read(parent / "contract.json")["contract"]["versions"])
    value = dict(identity=digest(body), contract=body)
    immutable(OUT / "contract.json", value)
    return value["identity"]


def verify_contract():
    assert base.verify_contract() == PARENT_ID
    value = read(OUT / "contract.json")
    assert digest(value["contract"]) == value["identity"]
    for path, checksum in value["contract"]["files"].items():
        assert sha(path) == checksum, path
    for package, version in value["contract"]["versions"].items():
        assert base.importlib.metadata.version(package) == version, package
    return value["identity"]


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
        np.testing.assert_array_equal(np.bincount(y[ids], minlength=3),
                                      np.bincount(y[original_ids], minlength=3))
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
    with (folder / "execution.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log), base.settings() as core, threadpool_limits(limits=1):
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
                print(base.json.dumps(dict(phase="loco", orientation=orientation, feature=feature)), flush=True)
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
                    worker_root=folder, call_key=f"{arm}_{model}_{orientation}_{block}",
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
    if repeat == 0:
        with np.load(base.OUT / scenario / f"{arm}_{model}.npz", allow_pickle=False) as parent:
            np.testing.assert_array_equal(profile, parent["orientations"])
        original_rows = pd.read_csv(base.OUT / scenario / f"{arm}_{model}.csv")
        pd.testing.assert_frame_equal(pd.concat(rows, ignore_index=True), original_rows, check_exact=False, rtol=1e-13, atol=1e-14)
    np.savez_compressed(folder / "profile.npz", orientations=profile, mean=profile.mean(axis=0))
    pd.concat(rows).to_csv(folder / "sources.csv", index=False)
    control_audit = {f: dict(parent=d["parent"], donor_hash=digest(d["donor_ids"].tolist()),
                            fit_levels=int(raw.iloc[fit][f].nunique()),
                            parent_fit_levels=int(raw.iloc[fit][d["parent"]].nunique())) for f, d in donors.items()}
    result = dict(identity=identity, key=key, scenario=scenario, model=model, arm=arm, repeat=repeat,
                  seconds=time.perf_counter()-started, source_rows=sum(len(t) for t in rows),
                  controls=control_audit, files={name: sha(folder / name) for name in ("profile.npz", "sources.csv", "roles.json")})
    immutable(folder / "complete.json", result)
    return dict(key=key, cached=False, seconds=result["seconds"])


def tasks(identity):
    return [(s, m, a, r, identity) for r in range(1, 13) for s in SCENARIOS
            for m in read(base.OUT / s / "prepared.json")["admitted"] for a in ("plain", "augmented")]


def status(**record):
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write(OUT / "progress.json", record)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as log:
        log.write(base.json.dumps(record) + "\n")
    print(base.json.dumps(record), flush=True)


def execute(identity, jobs, stage):
    all_tasks = tasks(identity)
    if stage == "preflight":
        all_tasks = [("additive", "lr", a, r, identity) for r in (0, 1) for a in ("plain", "augmented")]
    pending = [task for task in all_tasks if load_unit(task) is None]
    restored = len(all_tasks)-len(pending)
    costs = dict(lr=1, lgbm=2, xgb=3, catboost=4, et=5, rf=6, hgb=7)
    pending.sort(key=lambda t: (t[3], -costs[t[1]], t[0], t[2]))
    started = time.perf_counter()
    status(phase="fitting", total=len(all_tasks), restored=restored, pending=len(pending), jobs=jobs)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(worker, task) for task in pending]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status(phase="unit_complete", total=len(all_tasks), done=restored+n,
                   elapsed_seconds=round(time.perf_counter()-started, 2), **result)
    immutable(OUT / f"{stage}_complete.json", dict(identity=identity, units=len(all_tasks), status="complete"))


def evaluate(identity):
    assert read(OUT / "fit_complete.json")["identity"] == identity
    if (OUT / "results_complete.json").exists():
        return
    centers, graphs, counts = {}, {}, {}
    for scenario in SCENARIOS:
        admitted = read(base.OUT / scenario / "prepared.json")["admitted"]
        for arm in ("plain", "augmented"):
            repetitions = []
            for repeat in range(13):
                values = []
                for model in admitted:
                    task = (scenario, model, arm, repeat, identity)
                    if repeat:
                        assert load_unit(task) is not None
                        path = OUT / "units" / key_for(task) / "profile.npz"
                    else:
                        path = base.OUT / scenario / f"{arm}_{model}.npz"
                    with np.load(path, allow_pickle=False) as stored:
                        values.append(stored["mean"])
                repetitions.append(np.median(values, axis=0))
            centers[scenario, arm] = np.array(repetitions)
        for arm, gate, method in base.configurations():
            key = f"{scenario}__{arm}__{gate}__{method}"
            graphs[key], counts[key] = recurrence(centers[scenario, arm], arm, gate, method, adapter)
    with np.load(base.OUT / "graphs.npz", allow_pickle=False) as original:
        for key, graph in graphs.items():
            np.testing.assert_array_equal(graph[0], original[key])
            assert not (graph[1:] & ~graph[:-1]).any()
    np.savez_compressed(OUT / "profiles.npz", **{f"{s}__{a}": v for (s, a), v in centers.items()})
    np.savez_compressed(OUT / "graphs.npz", **graphs)
    np.savez_compressed(OUT / "counts.npz", **counts)
    immutable(OUT / "empirical_frozen.json", dict(identity=identity,
              files={name: sha(OUT / name) for name in ("profiles.npz", "graphs.npz", "counts.npz")}))
    rows, retention, edges = [], [], []
    with np.load(base.OUT / "references.npz", allow_pickle=False) as refs:
        for scenario in SCENARIOS:
            scores = refs[scenario]
            target = base.truth(scores)
            main_key = f"{scenario}__augmented__shadow__Joint4"
            anchor = graphs[main_key][0]
            for c, j, k in np.argwhere(anchor):
                edges.append(dict(scenario=scenario, class_id=int(c), leader=base.FEATURES[j],
                                  follower=base.FEATURES[k], correct=bool(target[c, j, k]),
                                  supports=int(counts[main_key][c, j, k])))
            for threshold in range(13):
                current = graphs[main_key][threshold]
                retention.append(dict(scenario=scenario, threshold=threshold,
                    retained_correct=int((current & target).sum()), retained_wrong=int((current & ~target).sum()),
                    lost_correct=int((anchor & ~current & target).sum()), lost_wrong=int((anchor & ~current & ~target).sum())))
            for arm, gate, method in base.configurations():
                key = f"{scenario}__{arm}__{gate}__{method}"
                for threshold, graph in enumerate(graphs[key]):
                    for target_name in (("common",) if method == "Joint4" else ("common", "own")):
                        own = SOURCES.index(method) if target_name == "own" else None
                        for c in range(3):
                            rows.append(dict(scenario=scenario, arm=arm, gate=gate, method=method,
                                target=target_name, class_id=c, threshold=threshold,
                                **base.grade(graph[c], scores[:, c], own)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "classes.csv", index=False)
    metrics = ["correct", "wrong", "declared", "true_total", "tied_wrong", "reverse_wrong", "disagreement_wrong"]
    for name, keys in (("by_scenario", ["scenario", "arm", "gate", "method", "target", "threshold"]),
                       ("overall", ["arm", "gate", "method", "target", "threshold"])):
        table = ratios(frame.groupby(keys)[metrics].sum().reset_index())
        table.to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(retention).to_csv(OUT / "main_retention.csv", index=False)
    pd.DataFrame(edges).to_csv(OUT / "main_edge_support.csv", index=False)
    assert verify_contract() == identity
    immutable(OUT / "results_complete.json", dict(identity=identity, scope="first_partition_twelve_record_resamples",
              independent_datasets=3, resamples_per_dataset=12, graphs=len(graphs)*13, class_rows=len(frame),
              files={p.name: sha(p) for p in OUT.iterdir() if p.suffix in (".csv", ".npz")}, status="complete"))
    status(phase="evaluated", graphs=len(graphs)*13, class_rows=len(frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "preflight", "fit", "evaluate", "all"), default="all")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock = OUT / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        identity = contract()
        verify_contract()
        for scenario in SCENARIOS:
            for repeat in range(13):
                roles_for(scenario, repeat)
        if args.stage in ("preflight", "all"):
            execute(identity, args.jobs, "preflight")
        if args.stage in ("fit", "all"):
            assert read(OUT / "preflight_complete.json")["identity"] == identity
            execute(identity, args.jobs, "fit")
        if args.stage in ("evaluate", "all"):
            evaluate(identity)
        status(phase="stage_complete", stage=args.stage)
    except BaseException as exc:
        write(OUT / f"failure_{time.time_ns()}.json", dict(error=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

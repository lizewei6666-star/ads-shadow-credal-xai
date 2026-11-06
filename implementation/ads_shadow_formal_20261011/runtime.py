"""ADS-specific orchestration around the audited numerical implementations."""
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
from types import FunctionType
import hashlib
import importlib.metadata
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(WORK / "ads_shadow_mixed_preflight_20261010"))
import adapter
import run_preflight as prior
sys.path.insert(0, str(WORK / "mixed_shadow_fit_repair_20261010"))
import engine as repair
from validation.execution import cached_selection
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
SOURCES = adapter.SOURCES
sha, digest = prior.sha, prior.digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior.write(path, value)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen object changed: {path}")
    else:
        write(path, value)


def bind(function, **dependencies):
    result = FunctionType(function.__code__, dict(function.__globals__, **dependencies),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


@contextmanager
def settings():
    core = adapter.get_core()
    budget = CONFIG["source_budget"]
    changes = dict(MODEL_N_JOBS=1, CANDIDATE_MODELS=CONFIG["models"], INNER_CV_FOLDS=5,
                   OUTPUT_FIGURES=False, ENABLE_CALIBRATION_DIAGNOSTICS=False,
                   SHAP_BACKGROUND_SIZE=budget["background"], SHAP_PERMUTATION_CYCLES=budget["shap_cycles"],
                   PERMUTATION_REPEATS=budget["pfi_repeats"],
                   TARGET_SHAP_PREDICTION_BATCH_ROWS=CONFIG["prediction_batch_rows"],
                   ENABLE_TARGET_SHAP_EQUIVALENCE_AUDIT=False,
                   ENABLE_TARGET_SHAP_PLAN_SENSITIVITY_AUDIT=False)
    before = {k: getattr(core, k) for k in changes}
    try:
        for k, v in changes.items():
            setattr(core, k, v)
        yield core
    finally:
        for k, v in before.items():
            setattr(core, k, v)


def data_roles(partition, repeat=0):
    raw, y, schema = prior.clean_data(adapter.get_core())
    roles = adapter.split_roles(adapter.partitions(y), y, partition-1, repeat)
    return raw, y, schema, roles


def freeze():
    raw, y, schema, _ = data_roles(1)
    parent_root = WORK / "mixed_shadow_pool_validation_20261010/outputs"
    parent_audit = read(parent_root / "independent_audit.json")
    assert parent_audit["status"] == "passed"
    dependencies = {ROOT / n for n in ("config.json", "PROTOCOL.md", "runtime.py", "run.py", "test_runtime.py")}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).resolve().is_relative_to(WORK):
            dependencies.add(Path(path).resolve())
    dependencies |= {parent_root / "independent_audit.json", parent_root / "results_complete.json",
                     WORK / "analysis_ads_credal_shadow_20260727_artifacts/run_meta.json"}
    body = dict(config=CONFIG, workbook_sha=adapter.WORKBOOK_SHA, rows=len(raw),
                classes=np.bincount(y, minlength=3).tolist(), features=schema.feature_cols,
                numeric=schema.numeric_cols, categorical=schema.categorical_cols,
                dependencies={str(p): sha(p) for p in sorted(dependencies)},
                versions={n: importlib.metadata.version(n) for n in
                          ("numpy", "pandas", "scipy", "scikit-learn", "shap", "xgboost", "lightgbm", "catboost")})
    value = dict(identity=digest(body), contract=body)
    immutable(OUT / "contract.json", value)
    return value["identity"]


def verify():
    value = read(OUT / "contract.json")
    assert digest(value["contract"]) == value["identity"]
    assert value["contract"]["config"] == CONFIG
    assert sha(adapter.WORKBOOK) == value["contract"]["workbook_sha"]
    for path, checksum in value["contract"]["dependencies"].items():
        assert sha(path) == checksum, path
    return value["identity"]


def selection_folder(b):
    return OUT / "admission" / f"partition_{b}"


def checked(path, identity):
    if not path.exists():
        return None
    value = read(path)
    assert value["identity"] == identity
    for name, checksum in value["files"].items():
        assert sha(path.parent / name) == checksum, path.parent / name
    return value


def prepare_partition(b):
    identity = verify()
    folder = selection_folder(b)
    saved = folder / "complete.json"
    existing = checked(saved, identity)
    if existing:
        return dict(partition=b, models=existing["admitted"], cached=True)
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    raw, y, schema, roles = data_roles(b)
    fit, cal, ext = roles
    selection_ids = np.concatenate([ids for i, ids in enumerate(adapter.partitions(y)) if i != b-1])
    assert set(selection_ids) == set(fit) | set(cal)
    part = adapter.get_core().OuterPartition(b, 1, b, 42, selection_ids, ext)
    started = time.perf_counter()
    with settings() as core, threadpool_limits(limits=1), (folder / "execution.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        selection, _, _ = cached_selection(core, raw, y, schema, [part], folder, digest([identity, b]))
        assert set(selection.model) == set(CONFIG["models"]) and selection.n_folds.eq(5).all()
        admitted = core.select_explanation_models_by_seed(selection, [b])[b]
    selection_cache = read(folder / "selection_cache.json")
    files = dict(selection_cache["artifacts"])
    write(folder / "roles.json", dict(zip(("fit", "calibration", "external"), [ids.tolist() for ids in roles])))
    for name in ("roles.json", "selection_cache.json"):
        files[name] = sha(folder / name)
    record = dict(identity=identity, partition=b, admitted=admitted,
                  best=str(selection.one_se_reference_model.iloc[0]), seconds=time.perf_counter()-started,
                  role_classes={name: np.bincount(y[ids], minlength=3).tolist() for name, ids in zip(("fit", "calibration", "external"), roles)}, files=files)
    immutable(saved, record)
    return dict(partition=b, models=admitted, cached=False, seconds=record["seconds"])


def admission(b):
    value = checked(selection_folder(b) / "complete.json", verify())
    if value is None:
        raise ValueError(f"Partition {b} admission is incomplete")
    return value


def task_key(task):
    b, model, arm, policy, r = task
    return f"p{b}__{model}__{arm}__{policy}__r{r:02d}"


def task_folder(task):
    return OUT / "units" / task_key(task)


def load_unit(task):
    return checked(task_folder(task) / "complete.json", verify())


def all_tasks():
    return [(b, m, arm, policy, r) for b in CONFIG["outer_partitions"] for m in admission(b)["admitted"]
            for arm in CONFIG["arms"] for policy in (CONFIG["policies"] if m == "lr" else ("shared",))
            for r in range(CONFIG["record_repeats"]+1)]


def tune(core, raw, y, schema, fit, arm, repeat, partition):
    construct = bind(repair.construct, adapter=adapter)
    bound = bind(repair.tune, FEATURES=list(schema.feature_cols), adapter=adapter, construct=construct)
    return bound(core, raw, y, schema, fit, arm, repeat, f"ads_partition_{partition}")


def compute(task):
    identity = verify()
    existing = load_unit(task)
    if existing:
        return dict(key=task_key(task), cached=True, seconds=existing["seconds"])
    b, model, arm, policy, r = task
    if policy not in ("fixed", "cv", "shared") or (model == "lr") == (policy == "shared"):
        raise ValueError("Invalid learner/policy combination")
    folder = task_folder(task)
    folder.mkdir(parents=True, exist_ok=True)
    raw, y, schema, roles = data_roles(b, r)
    fit, cal, ext = roles
    features = list(schema.feature_cols)
    work, ws, donors = repair.construct(raw, schema, fit, arm, r, b-1)
    write(folder / "roles.json", dict(zip(("fit", "calibration", "external"), [v.tolist() for v in roles])))
    profiles, frames, predictive, estimator_c = [], [], [], []
    then = time.perf_counter()
    with settings() as core, threadpool_limits(limits=1), (folder / "execution.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        if policy == "cv":
            c, table, tuning_roles = tune(core, raw, y, schema, fit, arm, r, b)
            table.to_csv(folder / "tuning.csv", index=False)
            write(folder / "tuning_roles.json", tuning_roles)
        else:
            c = 1.0
        with repair.regularized(core, c, estimator_c):
            for orientation in repair.orientations(arm):
                oriented = adapter.oriented(work, features, orientation) if arm == "double" else work
                train = oriented.iloc[fit].reset_index(drop=True)
                evaluation = oriented.iloc[np.r_[cal, ext]].reset_index(drop=True)
                source_seed = core.stable_seed_from_key("fixed_source_protocol_outer", b, model)
                estimator, encoded, probability, _ = core.fit_fixed_predictive_specification(
                    train, evaluation, y[fit], ws, model, core.stable_seed_from_key("fixed_spec_outer_fit", b, model), [0, 1, 2])
                check = prior.roundtrip(core, estimator, encoded, train, evaluation, ws, probability)
                reduced = {}
                for j, f in enumerate(ws.feature_cols):
                    reduced[f] = core.fit_loco_reduced_probability(
                        y_train=y[fit], X_train_raw=train, X_eval_raw=evaluation, schema=ws,
                        classes=[0, 1, 2], model_name=model, feature=f, seed=source_seed, n_jobs=1,
                        sample_weight=core.make_balanced_sample_weight(y[fit]))
                    print(json.dumps(dict(phase="loco", orientation=orientation, done=j+1, total=len(ws.feature_cols))), flush=True)
                part = core.OuterPartition(b, 1, b, 42, fit, ext)
                blocks = []
                for block, ids, segment in (("calibration", cal, slice(0, len(cal))), ("external", ext, slice(len(cal), None))):
                    print(json.dumps(dict(phase="sources_start", block=block, orientation=orientation)), flush=True)
                    ev, p = evaluation.iloc[segment].reset_index(drop=True), probability[segment]
                    tables, errors = core.compute_fitted_source_tables(
                        estimator=estimator, encoded=replace(encoded, X_test=encoded.X_test[segment]),
                        proba=p, y_train=y[fit], y_eval=y[ids], X_train_raw=train, X_eval_raw=ev,
                        schema=ws, class_order=[0, 1, 2], model_name=model, source_seed=source_seed,
                        feature_cols=ws.feature_cols, worker_root=folder, call_key=f"o{orientation}_{block}",
                        include_firm_table=True, shap_phase_scope="outer_actual", outer_partition=b,
                        inner_fold=-1, loco_reduced_proba_by_feature={f: a[segment] for f, a in reduced.items()})
                    if errors or set(tables) != set(SOURCES):
                        raise RuntimeError(f"Incomplete sources: {errors}")
                    table = pd.concat([core.standardize_outer_source_table(s, tables[s], part, model) for s in SOURCES], ignore_index=True)
                    assert len(table) == 12*len(ws.feature_cols) and np.isfinite(table.raw_statistic).all()
                    assert not table.duplicated(["source", "class_id", "feature"]).any()
                    assert all(t.n_eval.eq(len(ev)).all() for t in tables.values())
                    keyed = table.set_index(["source", "class_id", "feature"]).raw_statistic
                    names = adapter.aligned_names(features, ws, orientation) if arm == "double" else ws.feature_cols
                    blocks.append(np.array([keyed[s, cl, f] for s in SOURCES for cl in range(3) for f in names]).reshape(4, 3, -1))
                    frames.append(table[["source", "class_id", "feature", "raw_statistic"]].assign(block=block, orientation=orientation))
                    predictive.append(dict(block=block, orientation=orientation, n_eval=len(ids),
                        balanced_brier=repair.balanced_loss(core, y[ids], p),
                        balanced_accuracy=float(np.mean([(p.argmax(axis=1)[y[ids] == cl] == cl).mean() for cl in range(3)])),
                        roundtrip_error=check["max_probability_error"]))
                    print(json.dumps(dict(phase="sources_complete", block=block, orientation=orientation)), flush=True)
                profiles.append(blocks)
                del estimator, encoded, reduced
    values = np.asarray(profiles)
    assert values.shape == (len(repair.orientations(arm)), 2, 4, 3, len(ws.feature_cols))
    if model == "lr":
        assert len(estimator_c) == len(repair.orientations(arm))*(1+len(ws.feature_cols)) and set(estimator_c) == {c}
    np.savez_compressed(folder / "profile.npz", orientations=values, mean=values.mean(axis=0))
    pd.concat(frames, ignore_index=True).to_csv(folder / "sources.csv", index=False)
    pd.DataFrame(predictive).to_csv(folder / "predictive.csv", index=False)
    names = ["roles.json", "profile.npz", "sources.csv", "predictive.csv"]
    if policy == "cv":
        names += ["tuning.csv", "tuning_roles.json"]
    record = dict(identity=identity, task=list(task), C=c if model == "lr" else None, estimator_C=estimator_c,
                  seconds=time.perf_counter()-then, source_rows=sum(len(f) for f in frames),
                  controls={name: dict(parent=d["parent"], seed=d["seed"], donor_hash=digest(d["donor_ids"].tolist())) for name, d in donors.items()},
                  files={name: sha(folder/name) for name in names})
    assert verify() == identity
    immutable(folder / "complete.json", record)
    return dict(key=task_key(task), cached=False, seconds=record["seconds"])


def decisions(center, features, arm, gate, method):
    columns = features + [n for pair in adapter.control_names(features).values() for n in pair] if arm == "double" else features
    controls = adapter.control_positions(features, columns) if arm == "double" else np.full((len(features), 2), -1)
    votes = adapter.votes(center, len(features), controls, gate)
    if method == "Joint4":
        result = adapter.joint_graphs(votes)
    else:
        passed = votes[:, SOURCES.index(method)]
        anchor, counts = passed[0], passed[1:].sum(axis=0)
        result = dict(graphs=np.array([anchor & (counts >= k) for k in range(len(passed))]),
                      counts=counts, anchor=anchor, mask=np.ones_like(anchor))
    result["votes"] = votes
    return result


def profile(b, model, arm, policy, r):
    task = (b, model, arm, policy if model == "lr" else "shared", r)
    if load_unit(task) is None:
        raise ValueError(f"Missing unit: {task_key(task)}")
    with np.load(task_folder(task)/"profile.npz", allow_pickle=False) as stored:
        return stored["mean"].copy()


def weighted_profiles(values, sizes):
    sizes = np.asarray(sizes, dtype=float)
    if len(values) != len(sizes) or not len(sizes) or not (sizes > 0).all():
        raise ValueError("Invalid partition weights")
    return np.average(np.stack(values), axis=0, weights=sizes)


def evaluate():
    identity = verify()
    tasks = all_tasks()
    missing = [task_key(t) for t in tasks if load_unit(t) is None]
    if missing:
        raise ValueError(f"Final graphs require all {len(tasks)} units; {len(missing)} missing")
    features = read(OUT/"contract.json")["contract"]["features"]
    centers, scope_members = {}, {}
    sizes = [len(data_roles(b)[3][2]) for b in CONFIG["outer_partitions"]]
    for b in CONFIG["outer_partitions"]:
        selected = admission(b)
        models = selected["admitted"]
        scopes = dict(admitted_pool=models, best_learner=[selected["best"]])
        scopes.update({f"learner_{m}":[m] for m in models})
        if len(models) > 1:
            scopes.update({f"without_{m}":[n for n in models if n != m] for m in models})
        scope_members[str(b)] = scopes
        for arm in CONFIG["arms"]:
            for policy in CONFIG["policies"]:
                per_model = {m: np.stack([profile(b, m, arm, policy, r) for r in range(13)]) for m in models}
                for scope, members in scopes.items():
                    centers[f"p{b}__{arm}__{policy}__{scope}"] = np.median([per_model[m] for m in members], axis=0)
    for arm in CONFIG["arms"]:
        for policy in CONFIG["policies"]:
            for scope in ("admitted_pool", "best_learner"):
                key = f"{arm}__{policy}__{scope}"
                centers[f"combined__{key}"] = weighted_profiles([centers[f"p{b}__{key}"] for b in CONFIG["outer_partitions"]], sizes)
    graphs, counts, votes, summaries, relations, frontier = {}, {}, {}, [], [], []
    for key, center in centers.items():
        partition, arm, policy, scope = key.split("__")
        for gate in (("none", "zero", "shadow") if arm == "double" else ("none", "zero")):
            for method in (*SOURCES, "Joint4"):
                name = f"{key}__{gate}__{method}"
                result = decisions(center, features, arm, gate, method)
                graphs[name], counts[name] = result["graphs"], result["counts"]
                votes[name] = result["votes"]
                for k, graph in enumerate(result["graphs"]):
                    assert not (graph & graph.swapaxes(-1, -2)).any()
                    for c in range(3):
                        meta = dict(partition=partition, arm=arm, policy=policy, scope=scope, gate=gate, method=method, support=k, class_id=c)
                        undominated = ~graph[c].any(axis=0)
                        summaries.append(dict(**meta, declared=int(graph[c].sum()), undominated=int(undominated.sum()), unresolved=int(len(features)*(len(features)-1)//2-graph[c].sum())))
                        if k in (8, 10, 12):
                            for j, h in np.argwhere(graph[c]):
                                relations.append(dict(**meta, leader=features[j], follower=features[h], repetitions=int(result["counts"][c,j,h])))
                            for j in np.flatnonzero(undominated):
                                frontier.append(dict(**meta, feature=features[j]))
    for name, values in (("profiles", centers), ("graphs", graphs), ("counts", counts), ("votes", votes)):
        np.savez_compressed(OUT/f"{name}.npz", **values)
    write(OUT/"pooling_scopes.json", scope_members)
    pd.DataFrame(summaries).to_csv(OUT/"summary.csv", index=False)
    pd.DataFrame(relations).to_csv(OUT/"relations.csv", index=False)
    pd.DataFrame(frontier).to_csv(OUT/"undominated.csv", index=False)
    names = [f"{n}.npz" for n in ("profiles", "graphs", "counts", "votes")]+["summary.csv", "relations.csv", "undominated.csv", "pooling_scopes.json"]
    immutable(OUT/"results_complete.json", dict(identity=identity, units=len(tasks), empirical_precision=None,
        files={n:sha(OUT/n) for n in names}))
    return dict(status="complete", empirical_precision=None, units=len(tasks))

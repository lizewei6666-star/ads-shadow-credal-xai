"""Paired fit repair using training-only regularization selection."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(WORK / "mixed_shadow_confirmation_20261010"))
import runner as parent
from support import adapter, base, sample, np, pd, csv
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

OUT = ROOT / "outputs"
GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
ARMS = ("plain", "single", "double")
POLICIES = ("fixed", "cv")
FEATURES, SOURCES, SCENARIOS = parent.FEATURES, parent.SOURCES, parent.SCENARIOS
read, write, immutable, sha, digest = parent.read, parent.write, parent.immutable, parent.sha, parent.digest


@contextmanager
def regularized(core, c, calls=None):
    original = core.build_estimator

    def build(model_name, *args, **kwargs):
        estimator = original(model_name, *args, **kwargs)
        if model_name == "lr":
            estimator.set_params(C=float(c))
            if calls is not None:
                calls.append(float(estimator.C))
        return estimator

    core.build_estimator = build
    try:
        yield
    finally:
        core.build_estimator = original


def construct(raw, schema, fit, arm, repeat, partition=0):
    if arm == "plain":
        return raw, schema, {}
    assert arm in ("single", "double")
    result, expanded, donors = adapter.augment(raw, schema, fit, partition, repeat)
    if arm == "single":
        keep = list(schema.feature_cols) + [a for a, _ in adapter.control_names(schema.feature_cols).values()]
        expanded = adapter.get_core().FeatureSchema(keep, [f for f in expanded.numeric_cols if f in keep],
                                                    [f for f in expanded.categorical_cols if f in keep])
        result = result[list(raw.columns) + keep[len(schema.feature_cols):]].copy()
        donors = {f: d for f, d in donors.items() if f in keep}
    assert all(set(d["donor_ids"]) <= set(fit) for d in donors.values())
    return result, expanded, donors


def orientations(arm):
    return range(2 if arm == "double" else 1)


def roles_for(scenario, repeat):
    raw, y, schema = sample(scenario)
    roles = adapter.split_roles(adapter.partitions(y), y, 0, repeat)
    original = read(parent.POINT / scenario / "roles.json")
    for ids, name in zip(roles, ("fit", "calibration", "external")):
        assert set(ids) <= set(original[name])
        np.testing.assert_array_equal(np.bincount(y[ids], minlength=3), np.bincount(y[original[name]], minlength=3))
    return raw, y, schema, roles


def tuning_splits(fit, y, scenario, repeat):
    unique = np.unique(fit)
    assert np.bincount(y[unique], minlength=3).min() >= 3
    splitter = StratifiedKFold(3, shuffle=True, random_state=adapter.seed("fit_repair_cv_v1", scenario, repeat))
    result = []
    for train, valid in splitter.split(unique, y[unique]):
        ti, vi = fit[np.isin(fit, unique[train])], fit[np.isin(fit, unique[valid])]
        assert not (set(ti) & set(vi)) and len(ti)+len(vi) == len(fit)
        assert np.bincount(y[ti], minlength=3).min() > 0 and np.bincount(y[vi], minlength=3).min() > 0
        result.append((ti, vi))
    return result


def choose_c(table):
    scores = table.groupby("C").balanced_brier.mean().sort_index()
    assert tuple(scores.index) == GRID and np.isfinite(scores).all()
    return float(scores[scores <= scores.min()+1e-12].index.min())


def balanced_loss(core, y, p):
    return float(np.mean([core.class_balanced_one_vs_rest_brier(y, p[:, c], c) for c in range(3)]))


def tune(core, raw, y, schema, fit, arm, repeat, scenario):
    rows, audits = [], []
    for fold, (ti, vi) in enumerate(tuning_splits(fit, y, scenario, repeat)):
        work, ws, donors = construct(raw, schema, ti, arm, repeat, partition=100+fold)
        audits.append(dict(fold=fold, train=ti.tolist(), valid=vi.tolist(),
                           donors={f: dict(parent=d["parent"], seed=d["seed"], hash=digest(d["donor_ids"].tolist()))
                                   for f, d in donors.items()}))
        for orientation in orientations(arm):
            oriented = adapter.oriented(work, FEATURES, orientation) if arm == "double" else work
            train, ev = oriented.iloc[ti].reset_index(drop=True), oriented.iloc[vi].reset_index(drop=True)
            for c in GRID:
                with regularized(core, c):
                    estimator, _, prob, _ = core.fit_fixed_predictive_specification(
                        train, ev, y[ti], ws, "lr", core.stable_seed_from_key("fit_repair_cv", fold), [0, 1, 2])
                rows.append(dict(fold=fold, orientation=orientation, C=c, balanced_brier=balanced_loss(core, y[vi], prob)))
                del estimator
    table = pd.DataFrame(rows)
    return choose_c(table), table, audits


def compute_unit(scenario, arm, policy, repeat, folder):
    folder.mkdir(parents=True, exist_ok=True)
    raw, y, schema, (fit, cal, ext) = roles_for(scenario, repeat)
    work, ws, donors = construct(raw, schema, fit, arm, repeat)
    write(folder / "roles.json", dict(zip(("fit", "calibration", "external"), [v.tolist() for v in (fit, cal, ext)])))
    profiles, source_rows, predictive, estimator_c = [], [], [], []
    started = time.perf_counter()
    with (folder / "execution.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log), base.settings() as core, threadpool_limits(limits=1):
        if policy == "cv":
            c, tuning, cv_audit = tune(core, raw, y, schema, fit, arm, repeat, scenario)
            tuning.to_csv(folder / "tuning.csv", index=False)
            write(folder / "tuning_roles.json", cv_audit)
        else:
            assert policy == "fixed"
            c = 1.0
        with regularized(core, c, estimator_c):
            for orientation in orientations(arm):
                oriented = adapter.oriented(work, FEATURES, orientation) if arm == "double" else work
                train = oriented.iloc[fit].reset_index(drop=True)
                evaluation = oriented.iloc[np.r_[cal, ext]].reset_index(drop=True)
                source_seed = core.stable_seed_from_key("fixed_source_protocol_outer", 1, "lr")
                estimator, encoded, probability, _ = core.fit_fixed_predictive_specification(
                    train, evaluation, y[fit], ws, "lr", core.stable_seed_from_key("fixed_spec_outer_fit", 1, "lr"), [0, 1, 2])
                codec = core.RawFeatureCodec.fit(train, ws)
                np.testing.assert_array_equal(core.transform_raw(encoded.preprocessor, codec.decode(codec.encode(evaluation))), encoded.X_test)
                reduced = {f: core.fit_loco_reduced_probability(
                    y_train=y[fit], X_train_raw=train, X_eval_raw=evaluation, schema=ws,
                    classes=[0, 1, 2], model_name="lr", feature=f, seed=source_seed, n_jobs=1,
                    sample_weight=core.make_balanced_sample_weight(y[fit])) for f in ws.feature_cols}
                part = core.OuterPartition(1, 1, 1, 42, fit, ext)
                blocks = []
                for block, ids, segment in (("calibration", cal, slice(0, len(cal))), ("external", ext, slice(len(cal), None))):
                    ev = evaluation.iloc[segment].reset_index(drop=True)
                    p = probability[segment]
                    tables, errors = core.compute_fitted_source_tables(
                        estimator=estimator, encoded=replace(encoded, X_test=encoded.X_test[segment]),
                        proba=p, y_train=y[fit], y_eval=y[ids], X_train_raw=train, X_eval_raw=ev,
                        schema=ws, class_order=[0, 1, 2], model_name="lr", source_seed=source_seed,
                        feature_cols=ws.feature_cols, worker_root=folder, call_key=f"{arm}_{orientation}_{block}",
                        include_firm_table=True, shap_phase_scope="outer_actual", outer_partition=1,
                        inner_fold=-1, loco_reduced_proba_by_feature={f: v[segment] for f, v in reduced.items()})
                    assert not errors and set(tables) == set(SOURCES), errors
                    table = pd.concat([core.standardize_outer_source_table(s, tables[s], part, "lr") for s in SOURCES], ignore_index=True)
                    assert len(table) == 12*len(ws.feature_cols) and np.isfinite(table.raw_statistic).all()
                    assert not table.duplicated(["source", "class_id", "feature"]).any()
                    keyed = table.set_index(["source", "class_id", "feature"]).raw_statistic
                    order = adapter.aligned_names(FEATURES, ws, orientation) if arm == "double" else ws.feature_cols
                    blocks.append(np.array([keyed[s, cl, f] for s in SOURCES for cl in range(3) for f in order]).reshape(4, 3, -1))
                    source_rows.append(table[["source", "class_id", "feature", "raw_statistic"]].assign(block=block, orientation=orientation))
                    predictive.append(dict(block=block, orientation=orientation, balanced_brier=balanced_loss(core, y[ids], p),
                                           balanced_accuracy=float(np.mean([(p.argmax(axis=1)[y[ids] == cl] == cl).mean() for cl in range(3)]))))
                profiles.append(blocks)
                del estimator, encoded, reduced
    profile = np.asarray(profiles)
    assert profile.shape == (len(orientations(arm)), 2, 4, 3, len(ws.feature_cols))
    assert len(estimator_c) == len(orientations(arm))*(1+len(ws.feature_cols)) and set(estimator_c) == {c}
    np.savez_compressed(folder / "profile.npz", orientations=profile, mean=profile.mean(axis=0))
    pd.concat(source_rows, ignore_index=True).to_csv(folder / "sources.csv", index=False)
    pd.DataFrame(predictive).to_csv(folder / "predictive.csv", index=False)
    names = ["roles.json", "profile.npz", "sources.csv", "predictive.csv"]
    if policy == "cv":
        names += ["tuning.csv", "tuning_roles.json"]
    return dict(scenario=scenario, arm=arm, policy=policy, repeat=repeat, C=c, seconds=time.perf_counter()-started,
                estimator_C=estimator_c, source_rows=sum(len(t) for t in source_rows),
                donors={f: dict(parent=d["parent"], seed=d["seed"], hash=digest(d["donor_ids"].tolist())) for f, d in donors.items()},
                files={name: sha(folder / name) for name in names})


def gates(arm):
    return {"plain": ("none", "zero"), "single": ("none", "zero", "shadow"),
            "double": ("none", "zero", "shadow_A", "shadow")}[arm]


def decision_input(center, arm, gate):
    if arm == "plain":
        return center, "plain", gate
    if arm == "single":
        assert center.shape[-1] == 20
        center = np.concatenate([center[..., :10], np.repeat(center[..., 10:], 2, axis=-1)], axis=-1)
    if gate == "shadow_A":
        assert arm == "double"
        center = center.copy()
        center[..., 11::2] = center[..., 10::2]
    return center, "augmented", "shadow" if gate == "shadow_A" else gate


def decisions(center, arm, gate, method):
    adapted, mapped_arm, mapped_gate = decision_input(center, arm, gate)
    return parent.recurrence(adapted, mapped_arm, mapped_gate, method, adapter)

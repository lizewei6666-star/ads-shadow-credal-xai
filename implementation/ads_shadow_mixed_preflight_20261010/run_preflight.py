"""Run a bounded real-data execution audit without producing final priorities."""
from __future__ import annotations

from adapter import (ROOT, WORK, WORKBOOK, WORKBOOK_SHA, MODELS, SOURCES, get_core,
                     preflight_core, partitions, split_roles, augment, oriented,
                     aligned_names, control_names, seed)
import argparse
from contextlib import redirect_stdout, redirect_stderr
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(out, phase, **details):
    event = dict(phase=phase, **details)
    write(out / "progress.json", event)
    with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def clean_data(core):
    if sha(WORKBOOK) != WORKBOOK_SHA:
        raise RuntimeError("Workbook differs from the archived manuscript input")
    data = core.load_ads_table(str(WORKBOOK), use_hour_group=True)
    schema = core.get_feature_schema(data)
    raw, y = core.split_xy(data, schema)
    assert len(data) == 1469 and len(schema.feature_cols) == 14
    assert np.bincount(y, minlength=3).tolist() == [1328, 109, 32]
    old = json.loads((WORK / "analysis_ads_credal_shadow_20260727_artifacts/run_meta.json").read_text(encoding="utf-8"))
    assert schema.feature_cols == old["data"]["feature_columns"]
    assert schema.numeric_cols == old["data"]["numeric_columns"]
    assert schema.categorical_cols == old["data"]["categorical_columns"]
    # The original raw masker preserves these ADS string categories exactly.
    for feature in schema.categorical_cols:
        if not raw[feature].map(lambda x: isinstance(x, str)).all():
            raise ValueError(f"Native non-string categories need a separate codec audit: {feature}")
    if raw[schema.feature_cols].isna().any().any():
        raise ValueError("Unexpected missing cleaned predictors")
    return raw, y, schema


def probability_tolerance(native_dtype):
    if not np.issubdtype(native_dtype, np.floating):
        raise TypeError("Expected floating-point probability output")
    return float(max(1e-12, 8 * np.finfo(native_dtype).eps))


def roundtrip(core, estimator, encoded, train, evaluation, schema, probability):
    codec = core.RawFeatureCodec.fit(train, schema)
    transformed = core.transform_raw(encoded.preprocessor, codec.decode(codec.encode(evaluation)))
    np.testing.assert_array_equal(transformed, encoded.X_test)
    actual = core.predict_proba_aligned(estimator, transformed, np.array([0, 1, 2]))
    np.testing.assert_allclose(actual, probability, rtol=0, atol=1e-12)
    assert probability.shape == (len(evaluation), 3) and np.isfinite(probability).all()
    native_dtype = np.asarray(estimator.predict_proba(encoded.X_test[:1])).dtype
    normalization_tolerance = probability_tolerance(native_dtype)
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=normalization_tolerance)
    assert ((probability >= -1e-12) & (probability <= 1 + 1e-12)).all()
    unknown = {f: int((~evaluation[f].isin(train[f].unique())).sum()) for f in schema.categorical_cols}
    return dict(max_probability_error=float(np.max(abs(actual - probability))),
                native_probability_dtype=str(native_dtype),
                sum_to_one_max_error=float(abs(probability.sum(axis=1) - 1).max()),
                sum_to_one_tolerance=normalization_tolerance,
                unknown_category_rows={f: n for f, n in unknown.items() if n},
                encoded_columns=int(encoded.X_test.shape[1]))


def shap_audit(core, estimator, encoded, train, evaluation, labels, schema):
    selected = np.array([np.flatnonzero(labels == c)[0] for c in range(3)])
    probe = evaluation.iloc[selected]
    codec = core.RawFeatureCodec.fit(train, schema)
    bg_ids = core.sample_indices(len(train), 12, core._target_shap_background_seed("outer_actual", 1, -1))
    bg = codec.encode(train.iloc[bg_ids])
    x = codec.encode(probe)
    predict = core._make_raw_probability_function(estimator, encoded.preprocessor, codec, np.array([0, 1, 2]))
    p = len(schema.feature_cols)
    masks = [0, 1, (1 << p) - 1, 1 << (p - 1)]
    if p > 32:
        masks.append(1 << 32)
    keys = [(i, mask) for i in range(len(probe)) for mask in masks]
    computed, _ = core._evaluate_coalition_states(keys, x, bg, predict, 3, 4096)
    error = 0.0
    for (i, mask), value in computed.items():
        hybrid = train.iloc[bg_ids].copy()
        for j, feature in enumerate(schema.feature_cols):
            if mask & (1 << j):
                hybrid[feature] = probe.iloc[i][feature]
        expected = core.predict_proba_aligned(estimator, core.transform_raw(encoded.preprocessor, hybrid), np.array([0, 1, 2])).mean(axis=0)
        np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)
        error = max(error, float(abs(value - expected).max()))
    args = dict(estimator=estimator, preprocessor=encoded.preprocessor, codec=codec,
                X_eval_mask=x, background=bg,
                stable_record_ids=probe[core.STABLE_RECORD_ID_COL].to_numpy(),
                classes=np.array([0, 1, 2]), plan_version=core.TARGET_SHAP_PERMUTATION_PLAN_VERSION,
                phase_scope="outer_actual", outer_partition=1, inner_fold=-1,
                full_path_max_evals=2 * p + 1, prediction_batch_rows=4096)
    target, _ = core._target_coordinate_shap_values(**args, target_positions=list(range(p)))
    reference, _ = core._full_reference_shap_values(**args)
    np.testing.assert_allclose(target, reference, rtol=0, atol=1e-12)
    residual = target.sum(axis=1) - (predict(x) - predict(bg).mean(axis=0))
    np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-12)
    return dict(coalition_checks=len(keys), direct_max_error=error,
                complete_path_max_error=float(abs(target - reference).max()),
                additive_max_error=float(abs(residual).max()), maximum_mask=max(masks))


def explanation_unit(core, out, raw, schema, features, y, roles, arm, orientation):
    fit, cal, external = roles
    evaluation_ids = np.concatenate([cal, external])
    working = oriented(raw, features, orientation) if arm == "augmented" else raw
    train, evaluation = working.iloc[fit].reset_index(drop=True), working.iloc[evaluation_ids].reset_index(drop=True)
    fit_seed = core.stable_seed_from_key("fixed_spec_outer_fit", 1, "lr")
    source_seed = core.stable_seed_from_key("fixed_source_protocol_outer", 1, "lr")
    estimator, encoded, probability, _ = core.fit_fixed_predictive_specification(train, evaluation, y[fit], schema, "lr", fit_seed, [0, 1, 2])
    checks = dict(roundtrip=roundtrip(core, estimator, encoded, train, evaluation, schema, probability),
                  shap=shap_audit(core, estimator, encoded, train, evaluation, y[evaluation_ids], schema))
    reduced = {}
    for j, feature in enumerate(schema.feature_cols):
        reduced[feature] = core.fit_loco_reduced_probability(
            y_train=y[fit], X_train_raw=train, X_eval_raw=evaluation, schema=schema,
            classes=[0, 1, 2], model_name="lr", feature=feature, seed=source_seed,
            n_jobs=1, sample_weight=core.make_balanced_sample_weight(y[fit]))
        if j % 7 == 0 or j + 1 == len(schema.feature_cols):
            progress(out, "loco", arm=arm, orientation=orientation, done=j + 1, total=len(schema.feature_cols))
    repeat_feature = schema.feature_cols[-1]
    again = core.fit_loco_reduced_probability(y_train=y[fit], X_train_raw=train,
        X_eval_raw=evaluation, schema=schema, classes=[0, 1, 2], model_name="lr",
        feature=repeat_feature, seed=source_seed, n_jobs=1,
        sample_weight=core.make_balanced_sample_weight(y[fit]))
    np.testing.assert_allclose(again, reduced[repeat_feature], rtol=0, atol=1e-12)
    checks["loco_refit_max_error"] = float(abs(again - reduced[repeat_feature]).max())
    frames, profiles, block_checks = [], [], []
    part = core.OuterPartition(1, 1, 1, 42, fit, external)
    for stage, segment in (("calibration", slice(0, len(cal))), ("external", slice(len(cal), None))):
        ev = evaluation.iloc[segment].reset_index(drop=True)
        lab = y[evaluation_ids][segment]
        prob = probability[segment]
        tables, errors = core.compute_fitted_source_tables(
            estimator=estimator, encoded=replace(encoded, X_test=encoded.X_test[segment]),
            proba=prob, y_train=y[fit], y_eval=lab, X_train_raw=train, X_eval_raw=ev,
            schema=schema, class_order=[0, 1, 2], model_name="lr", source_seed=source_seed,
            feature_cols=schema.feature_cols, worker_root=out, call_key=f"{arm}_{orientation}_{stage}",
            include_firm_table=True, shap_phase_scope="outer_actual", outer_partition=1,
            inner_fold=-1, loco_reduced_proba_by_feature={f: a[segment] for f, a in reduced.items()})
        if errors or set(tables) != set(SOURCES):
            raise RuntimeError(f"Incomplete source execution: {errors}")
        reference = core.compute_class_permutation_importance_reference(
            estimator, encoded.preprocessor, ev, lab, [0, 1, 2], schema.feature_cols,
            "lr", source_seed, repeats=3, base_proba=prob)
        index = ["class_id", "feature"]
        observed = tables["PFI"].set_index(index).balanced_mean_brier_increase.sort_index()
        expected = reference.set_index(index).balanced_mean_brier_increase.sort_index()
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
        table = pd.concat([core.standardize_outer_source_table(s, tables[s], part, "lr") for s in SOURCES], ignore_index=True)
        coverage = set(zip(table.source, table.class_id, table.feature))
        assert coverage == {(s, c, f) for s in SOURCES for c in range(3) for f in schema.feature_cols}
        assert len(table) == 4 * 3 * len(schema.feature_cols)
        assert np.isfinite(table.raw_statistic).all()
        assert all(t.n_eval.eq(len(ev)).all() for t in tables.values())
        keyed = table.set_index(["source", "class_id", "feature"]).raw_statistic
        loco_error, firm_error = 0.0, 0.0
        for c in range(3):
            w = np.where(lab == c, .5 / (lab == c).sum(), .5 / (lab != c).sum())
            target = (lab == c).astype(float)
            for f in schema.feature_cols:
                contrast = float(np.sum(w * ((target - reduced[f][segment, c])**2 - (target - prob[:, c])**2)))
                loco_error = max(loco_error, abs(contrast - keyed["LOCO", c, f]))
                means, masses = [], []
                for level in ev[f].unique():
                    selected = (ev[f] == level).to_numpy()
                    mass = w[selected].sum()
                    means.append(np.sum(w[selected] * prob[selected, c]) / mass)
                    masses.append(mass)
                means, masses = np.array(means), np.array(masses)
                firm = np.sqrt(np.sum(masses * (means - np.sum(masses * means))**2))
                firm_error = max(firm_error, abs(float(firm) - keyed["FIRM", c, f]))
        assert loco_error < 1e-12 and firm_error < 1e-12
        order = aligned_names(features, schema, orientation) if arm == "augmented" else schema.feature_cols
        profiles.append(np.array([keyed[s, c, f] for s in SOURCES for c in range(3) for f in order]).reshape(4, 3, len(order)))
        frames.append(table[["source", "class_id", "feature", "raw_statistic"]].assign(block=stage, arm=arm, orientation=orientation))
        block_checks.append(dict(block=stage, n_eval=len(ev), source_rows=len(table),
            pfi_reference_max_error=float(abs(observed - expected).max()),
            independent_loco_max_error=loco_error, independent_firm_max_error=firm_error))
        progress(out, "sources_complete", arm=arm, orientation=orientation, block=stage, rows=len(table))
    pd.concat(frames).to_csv(out / f"scores_{arm}_o{orientation}.csv", index=False)
    np.save(out / f"profile_{arm}_o{orientation}.npy", np.array(profiles), allow_pickle=False)
    return dict(**checks, blocks=block_checks, n_full_fits=1, n_reduced_fits=len(reduced) + 1,
                rows=sum(b["source_rows"] for b in block_checks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Preflight output must stay inside its dedicated directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Already completed; inspect audit instead of rerunning")
    lock = out / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.perf_counter()
    try:
        core = get_core()
        raw, y, schema = clean_data(core)
        folds = partitions(y)
        roles = split_roles(folds, y, 0)
        augmented, augmented_schema, donors = augment(raw, schema, roles[0], 0)
        code = {str(p): sha(p) for p in [ROOT / "adapter.py", ROOT / "run_preflight.py", ROOT / "PROTOCOL.md", Path(core.__file__)]}
        contract = dict(workbook_sha256=WORKBOOK_SHA, code=code, feature_order=schema.feature_cols,
                        roles={name: dict(n=len(ids), classes=np.bincount(y[ids], minlength=3).tolist(), digest=digest(ids.tolist()))
                               for name, ids in zip(("fit", "calibration", "external"), roles)},
                        models=list(MODELS), explanation_model="lr", explanation_budgets=dict(background=12, cycles=1, pfi=3),
                        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "shap", "xgboost", "lightgbm", "catboost")})
        identity = digest(contract)
        manifest = dict(identity=identity, contract=contract)
        if (out / "manifest.json").exists():
            assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest, "Changed execution contract"
        else:
            write(out / "manifest.json", manifest)
        feature_rows = []
        for f in schema.feature_cols:
            levels = set(raw.iloc[roles[0]][f])
            feature_rows.append(dict(feature=f, kind="numeric" if f in schema.numeric_cols else "categorical",
                all_levels=raw[f].nunique(), fit_levels=len(levels),
                calibration_unseen=int((~raw.iloc[roles[1]][f].isin(levels)).sum()),
                external_unseen=int((~raw.iloc[roles[2]][f].isin(levels)).sum())))
        pd.DataFrame(feature_rows).to_csv(out / "feature_audit.csv", index=False)
        donor_summary = []
        for control, record in donors.items():
            assert set(record["donor_ids"]) <= set(roles[0])
            donor_summary.append(dict(control=control, parent=record["parent"], seed=record["seed"],
                donor_digest=digest(record["donor_ids"].tolist()),
                fitting_levels=int(augmented.iloc[roles[0]][control].nunique()),
                inherited_kind="numeric" if record["parent"] in schema.numeric_cols else "categorical"))
        pd.DataFrame(donor_summary).to_csv(out / "control_audit.csv", index=False)
        checks = []
        for b in range(5):
            for r in range(13):
                fit, cal, ext = split_roles(folds, y, b, r)
                sampled, ss, dd = augment(raw, schema, fit, b, r)
                assert len(ss.feature_cols) == 42
                assert all(set(d["donor_ids"]) <= set(fit) for d in dd.values())
                checks.append(dict(partition=b + 1, repeat=r, fit=len(fit), calibration=len(cal), external=len(ext)))
        write(out / "construction_checks.json", dict(status="passed", role_control_cases=len(checks), cases=checks))
        progress(out, "prepared", roles=contract["roles"], raw_features=14, augmented_features=42)
        completed = []
        with threadpool_limits(limits=1), preflight_core():
            for arm, frame, spec in (("plain", raw, schema), ("augmented", augmented, augmented_schema)):
                fit, cal, ext = roles
                train = frame.iloc[fit].reset_index(drop=True)
                evaluation = frame.iloc[np.r_[cal, ext]].reset_index(drop=True)
                for model in MODELS:
                    key = f"compatibility_{arm}_{model}"
                    result_path = out / f"{key}.json"
                    if result_path.exists():
                        record = json.loads(result_path.read_text(encoding="utf-8"))
                        assert record["identity"] == identity
                        completed.append(record)
                        continue
                    then = time.perf_counter()
                    progress(out, "compatibility_start", arm=arm, model=model)
                    with (out / f"{key}.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                        estimator, encoded, prob, _ = core.fit_fixed_predictive_specification(train, evaluation, y[fit], spec, model,
                            core.stable_seed_from_key("fixed_spec_outer_fit", 1, model), [0, 1, 2])
                        check = roundtrip(core, estimator, encoded, train, evaluation, spec, prob)
                    record = dict(identity=identity, key=key, status="passed", seconds=time.perf_counter() - then, checks=check)
                    write(result_path, record)
                    completed.append(record)
                    progress(out, "compatibility_complete", arm=arm, model=model, seconds=round(record["seconds"], 2))
                    del estimator, encoded, prob
            for arm, frame, spec, orientation in (("plain", raw, schema, 0), ("augmented", augmented, augmented_schema, 0),
                                                   ("augmented", augmented, augmented_schema, 1)):
                key = f"explanations_{arm}_o{orientation}"
                result_path = out / f"{key}.json"
                if result_path.exists():
                    record = json.loads(result_path.read_text(encoding="utf-8"))
                    assert record["identity"] == identity
                    for name, h in record["files"].items():
                        assert sha(out / name) == h
                else:
                    then = time.perf_counter()
                    progress(out, "explanations_start", arm=arm, orientation=orientation)
                    with (out / f"{key}.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                        check = explanation_unit(core, out, frame, spec, schema.feature_cols, y, roles, arm, orientation)
                    files = [out / f"scores_{arm}_o{orientation}.csv", out / f"profile_{arm}_o{orientation}.npy"]
                    record = dict(identity=identity, key=key, status="passed", seconds=time.perf_counter() - then,
                                  checks=check, files={p.name: sha(p) for p in files})
                    write(result_path, record)
                completed.append(record)
                progress(out, "explanations_complete", arm=arm, orientation=orientation, seconds=round(record["seconds"], 2))
        for p, h in code.items():
            assert sha(p) == h, f"Source changed: {p}"
        assert sha(WORKBOOK) == WORKBOOK_SHA
        for arm, count in (("plain", 1), ("augmented", 2)):
            values = [np.load(out / f"profile_{arm}_o{i}.npy", allow_pickle=False) for i in range(count)]
            np.save(out / f"profile_{arm}_orientation_mean.npy", np.mean(values, axis=0), allow_pickle=False)
        report = dict(status="passed", identity=identity, scope="execution_preflight_only", records=1469,
                      scientific_features=14, matched_controls=28, compatibility_units=14,
                      explanation_units=3, evaluated_source_rows=2352, bootstrap_fit_replicates=0,
                      final_relation_graph=False, empirical_precision=None,
                      seconds=time.perf_counter() - started, units=completed,
                      files={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".csv", ".npy") and p.name not in ("progress.json", "failure.json")})
        write(out / "complete.json", report)
        progress(out, "complete", seconds=round(report["seconds"], 2), empirical_precision=None)
    except BaseException as exc:
        write(out / "failure.json", dict(status="failed", error=repr(exc), seconds=time.perf_counter() - started))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

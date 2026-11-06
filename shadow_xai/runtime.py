"""Portable execution of the manuscript's frozen numerical workers."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "implementation"
STATE = {}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def check_environment():
    if sys.flags.optimize:
        raise RuntimeError("Do not use python -O: numerical audit assertions must remain enabled")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Fitting requires Python 3.10; use the supplied environment.yml")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line:
            package, expected = line.split("==")
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"{package}: expected {expected}, found {actual}")


@lru_cache(maxsize=1)
def modules():
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    for path in (IMPLEMENTATION, IMPLEMENTATION / "ads-shadow-credal-xai",
                 IMPLEMENTATION / "ads_shadow_formal_20261011"):
        sys.path.insert(0, str(path))
    runtime = importlib.import_module("runtime")
    support = importlib.import_module("support")
    return runtime, support


def verify():
    stored = read(STATE["out"] / "run_manifest.json")
    if stored != STATE["manifest"]:
        raise RuntimeError("Run manifest changed during execution")
    return stored["identity"]


def progress(phase, **details):
    value = dict(phase=phase, time=time.strftime("%Y-%m-%d %H:%M:%S"), **details)
    write(STATE["out"] / "progress.json", value)
    with (STATE["out"] / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value)+"\n")
    print(json.dumps(value), flush=True)


def fresh(case):
    _, support = modules()
    seed, scenario = case.split("-", 1)
    return support.sample(scenario, master=int(seed))


def simulation_roles(case, repeat):
    runtime, _ = modules()
    raw, y, schema = fresh(case)
    return raw, y, schema, runtime.adapter.split_roles(runtime.adapter.partitions(y), y, 0, repeat)


def simulation_tuning(core, raw, y, schema, fit, arm, repeat, case):
    runtime, _ = modules()
    return runtime.repair.tune(core, raw, y, schema, fit, arm, repeat, case.split("-", 1)[1])


def ads_data(core):
    import numpy as np
    expected = read(ROOT / "configs/ads_schema.json")
    path = Path(STATE["data"])
    if sha(path) != expected["workbook_sha256"]:
        raise ValueError("Workbook differs from the manuscript input; use the original private workbook")
    frame = core.load_ads_table(str(path), use_hour_group=True)
    schema = core.get_feature_schema(frame)
    raw, y = core.split_xy(frame, schema)
    assert len(raw) == expected["n_records"]
    assert np.bincount(y, minlength=3).tolist() == expected["class_counts"]
    for attr, key in (("feature_cols", "feature_columns"), ("numeric_cols", "numeric_columns"),
                      ("categorical_cols", "categorical_columns")):
        assert list(getattr(schema, attr)) == expected[key]
    assert not raw[schema.feature_cols].isna().any().any()
    assert all(raw[f].map(lambda x: isinstance(x, str)).all() for f in schema.categorical_cols)
    return raw, y, schema


def configure(kind, out, data=None):
    check_environment()
    out = Path(out).resolve()
    config = read(ROOT / "configs" / f"{kind}.json")
    if config["record_repeats"] != 12 or config["primary_support"] != 8:
        raise ValueError("The manuscript reporting protocol requires 12 resamples and primary support 8")
    if kind == "simulation":
        expected_budget = dict(background=12, shap_cycles=1, pfi_repeats=3)
        if (config["n_samples"], config["n_features"], config["source_budget"]) != (1500, 10, expected_budget):
            raise ValueError("The frozen simulation requires 1500 records, 10 features, and the published source budget")
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
               for folder in ("implementation", "shadow_xai") for p in sorted((ROOT/folder).rglob("*.py"))}
    body = dict(kind=kind, config=config, sources=sources,
                versions={line.split("==")[0]: line.split("==")[1]
                          for line in (ROOT/"requirements.txt").read_text().splitlines() if "==" in line})
    if kind == "ads":
        if data is None or not Path(data).is_file():
            raise FileNotFoundError("Supply --data with the private ADS workbook path")
        body["data_sha256"] = sha(data)
        if body["data_sha256"] != read(ROOT / "configs/ads_schema.json")["workbook_sha256"]:
            raise ValueError("Workbook differs from the manuscript input")
    identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    manifest = dict(identity=identity, contract=body)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise ValueError("Inputs, configuration or code changed. Choose a new output directory")
    elif any(p.name != "run.lock" for p in out.iterdir()):
        raise ValueError("Nonempty output directory has no matching run manifest")
    else:
        write(path, manifest)
    STATE.update(kind=kind, out=out, data=data, manifest=manifest, config=config)
    runtime, support = modules()
    if kind == "ads":
        runtime.OUT = out
        runtime.CONFIG = config
        runtime.verify = verify
        runtime.prior.clean_data = ads_data
        raw, y, schema = ads_data(runtime.adapter.get_core())
        contract = dict(identity=identity, contract=dict(features=list(schema.feature_cols)))
        if (out / "contract.json").exists():
            if read(out / "contract.json") != contract:
                raise ValueError("ADS feature contract changed")
        else:
            write(out / "contract.json", contract)
    return runtime, support


@contextmanager
def run_lock(folder):
    """The OS releases this lock after interruption or a machine restart."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process is using this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def simulation_bindings():
    _, support = modules()
    return support.bindings(verify, progress, out=STATE["out"],
                            point=STATE["out"] / "admission", generator=fresh)


def simulation_folder(task):
    case, model, arm, policy, repeat = task
    if model == "lr":
        return STATE["out"] / "lr_units" / f"{case}__{arm}__{policy}__r{repeat:02d}"
    _, support = modules()
    if repeat:
        old_arm = "augmented" if arm == "double" else arm
        key = support.previous.key_for((case, model, old_arm, repeat, verify()))
        return STATE["out"] / "units" / key
    return STATE["out"] / "admission" / case


def simulation_profile(task):
    case, model, arm, policy, repeat = task
    folder = simulation_folder(task)
    if model == "lr" or repeat:
        return folder / "profile.npz"
    return folder / f"{'augmented' if arm == 'double' else arm}_{model}.npz"


def checked_unit(task):
    case, model, arm, policy, repeat = task
    folder = simulation_folder(task)
    name = "complete.json" if model == "lr" or repeat else f"{'augmented' if arm == 'double' else arm}_{model}.json"
    path = folder / name
    if not path.exists():
        return None
    record = read(path)
    if record["identity"] != verify():
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    for name, checksum in record["files"].items():
        if sha(folder / name) != checksum:
            raise ValueError(f"Checkpoint file changed: {folder / name}")
    return record


def compute_simulation(task):
    runtime, support = modules()
    record = checked_unit(task)
    if record is not None:
        return dict(task=task, cached=True)
    case, model, arm, policy, repeat = task
    if model == "lr":
        worker = support.bind(runtime.repair.compute_unit, roles_for=simulation_roles, tune=simulation_tuning)
        record = worker(case, arm, policy, repeat, simulation_folder(task))
        record["identity"] = verify()
        write(simulation_folder(task) / "complete.json", record)
    else:
        bound = simulation_bindings()
        old_arm = "augmented" if arm == "double" else arm
        if repeat:
            bound.repeat_worker((case, model, old_arm, repeat, verify()))
        else:
            bound.point_worker((case, model, old_arm, verify()))
    if checked_unit(task) is None:
        raise RuntimeError("Worker returned without a complete checkpoint")
    return dict(task=task, cached=False)


def prepare():
    runtime, support = modules()
    if STATE["kind"] == "ads":
        for b in STATE["config"]["outer_partitions"]:
            progress("admission_complete", **runtime.prepare_partition(b))
    else:
        bound = simulation_bindings()
        for case in STATE["config"]["cases"]:
            bound.prepare(case, verify())


def tasks():
    runtime, _ = modules()
    if STATE["kind"] == "ads":
        return runtime.all_tasks()
    cfg = STATE["config"]
    return [(case, model, arm, policy, repeat)
            for case in cfg["cases"]
            for model in read(STATE["out"] / "admission" / case / "prepared.json")["admitted"]
            for arm in cfg["arms"]
            for policy in (cfg["policies"] if model == "lr" else ["shared"])
            for repeat in range(cfg["record_repeats"]+1)]


def worker(payload):
    kind, output, data, task = payload
    runtime, _ = configure(kind, output, data)
    return runtime.compute(tuple(task)) if kind == "ads" else compute_simulation(tuple(task))


def evaluate_simulation():
    import numpy as np
    import pandas as pd
    from .evidence import summarize
    _, support = modules()
    cfg, out = STATE["config"], STATE["out"]
    profiles, references, scopes = {}, {}, {}
    for case in cfg["cases"]:
        models = read(out / "admission" / case / "prepared.json")["admitted"]
        best = str(pd.read_pickle(out / "admission" / case / "selection.pkl").one_se_reference_model.iloc[0])
        members = dict(admitted_pool=models, best_learner=[best], **{f"learner_{m}": [m] for m in models})
        if len(models) > 1:
            members.update({f"without_{m}": [n for n in models if n != m] for m in models})
        scopes[case] = members
        for arm in cfg["arms"]:
            for policy in cfg["policies"]:
                values = {}
                for m in models:
                    records = []
                    for repeat in range(cfg["record_repeats"]+1):
                        task = (case, m, arm, policy if m == "lr" else "shared", repeat)
                        if checked_unit(task) is None:
                            raise ValueError(f"Missing unit: {task}")
                        with np.load(simulation_profile(task), allow_pickle=False) as stored:
                            records.append(stored["mean"].copy())
                    values[m] = np.stack(records)
                for scope, subset in members.items():
                    profiles[f"{case}__{'augmented' if arm == 'double' else arm}__{policy}__{scope}"] = np.median([values[m] for m in subset], axis=0)
        _, y, _ = fresh(case)
        fit = read(out / "admission" / case / "roles.json")["fit"]
        references[case] = support.design.population_scores(case.split("-", 1)[1], np.bincount(y[fit], minlength=3)/len(fit))
    np.savez_compressed(out / "profiles.npz", **profiles)
    np.savez_compressed(out / "references.npz", **references)
    write(out / "pooling_scopes.json", scopes)
    return summarize(out / "profiles.npz", out / "tables", "simulation", references=out / "references.npz")

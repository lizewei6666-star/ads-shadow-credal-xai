"""Prespecified fresh-sample confirmation with unchanged numerical workers."""
from __future__ import annotations

from support import (ROOT, OUT, POINT, DISCOVERY, MASTER, PRIMARY, SENSITIVITY,
                     previous, base, design, adapter, SCENARIOS, SOURCES, FEATURES,
                     sha, digest, read, write, immutable, sample, bindings,
                     recurrence, ratios, np, pd)
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time


def contract():
    discovery_identity = previous.verify_contract()
    paths = [ROOT / name for name in ("PROTOCOL.md", "support.py", "runner.py", "test_confirmation.py")]
    paths += [DISCOVERY / "outputs" / name for name in ("contract.json", "independent_audit.json", "results_complete.json")]
    body = dict(version="fresh_mixed_confirmation_v1", master_seed=MASTER,
                discovery_identity=discovery_identity, n=1500, p=10,
                scenarios=list(SCENARIOS), partitions=[1], repeats=list(range(13)),
                primary=PRIMARY, sensitivity=list(SENSITIVITY),
                files={str(p): sha(p) for p in paths},
                versions=read(DISCOVERY / "outputs/contract.json")["contract"]["versions"])
    value = dict(identity=digest(body), contract=body)
    immutable(OUT / "contract.json", value)
    return value["identity"]


def verify_contract():
    value = read(OUT / "contract.json")
    assert digest(value["contract"]) == value["identity"]
    assert previous.verify_contract() == value["contract"]["discovery_identity"]
    for name, checksum in value["contract"]["files"].items():
        assert sha(name) == checksum, name
    for package, version in value["contract"]["versions"].items():
        assert base.importlib.metadata.version(package) == version
    return value["identity"]


def status(**record):
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write(OUT / "progress.json", record)
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as log:
        log.write(base.json.dumps(record)+"\n")
    print(base.json.dumps(record), flush=True)


def prepare_status(phase, **detail):
    status(phase=phase, **detail)


def verify_inputs():
    value = read(OUT / "input_manifest.json")
    assert value["identity"] == verify_contract()
    for path, checksum in value["files"].items():
        assert sha(path) == checksum, path
    return value["identity"]


def prepare(identity):
    bound = bindings(verify_inputs, prepare_status)
    files = {}
    records = {}
    for scenario in SCENARIOS:
        record = bound.prepare(scenario, identity)
        assert record["identity"] == identity
        records[scenario] = dict(admitted=record["admitted"], classes=record["classes"], role_classes=record["role_classes"])
        folder = POINT / scenario
        paths = [folder / "prepared.json"] + [folder / name for name in record["files"]]
        paths.extend(folder / name for name in read(folder / "selection_cache.json")["artifacts"])
        files.update({str(p): sha(p) for p in paths})
        for repeat in range(13):
            bound.roles(scenario, repeat)
    count = 26*sum(len(v["admitted"]) for v in records.values())
    immutable(OUT / "input_manifest.json", dict(identity=identity, master_seed=MASTER, samples=records, files=files, units=count))
    status(phase="prepared", units=count, samples=records)


def key_for(task):
    return previous.key_for(task)


def unit_record(task):
    scenario, model, arm, repeat, identity = task
    if repeat:
        folder = OUT / "units" / key_for(task)
        path = folder / "complete.json"
    else:
        folder = POINT / scenario
        path = folder / f"{arm}_{model}.json"
    if not path.exists():
        return None
    record = read(path)
    assert record["identity"] == identity
    assert (record["scenario"], record["model"], record["arm"]) == (scenario, model, arm)
    if repeat:
        assert record["repeat"] == repeat and record["key"] == key_for(task)
    for name, checksum in record["files"].items():
        assert sha(folder / name) == checksum, (key_for(task), name)
    return record


def profile_path(task):
    scenario, model, arm, repeat, _ = task
    return OUT / "units" / key_for(task) / "profile.npz" if repeat else POINT / scenario / f"{arm}_{model}.npz"


def source_path(task):
    scenario, model, arm, repeat, _ = task
    return OUT / "units" / key_for(task) / "sources.csv" if repeat else POINT / scenario / f"{arm}_{model}.csv"


def tasks(identity):
    return [(s, m, a, r, identity) for r in range(13) for s in SCENARIOS
            for m in read(POINT / s / "prepared.json")["admitted"] for a in ("plain", "augmented")]


def worker(task):
    scenario, model, arm, repeat, identity = task
    assert verify_inputs() == identity
    record = unit_record(task)
    if record:
        return dict(key=key_for(task), cached=True, seconds=record["seconds"])
    bound = bindings(verify_inputs, prepare_status)
    if repeat:
        result = bound.repeat_worker(task)
    else:
        result = bound.point_worker((scenario, model, arm, identity))
    assert unit_record(task) is not None
    return dict(key=key_for(task), cached=result["cached"], seconds=result["seconds"])


def execute(identity, jobs, stage):
    all_tasks = tasks(identity)
    if stage == "preflight":
        model = read(POINT / SCENARIOS[0] / "prepared.json")["admitted"][0]
        all_tasks = [(SCENARIOS[0], model, a, r, identity) for r in (0, 1) for a in ("plain", "augmented")]
    pending = [t for t in all_tasks if unit_record(t) is None]
    restored = len(all_tasks)-len(pending)
    costs = dict(lr=1, lgbm=2, xgb=3, catboost=4, et=5, rf=6, hgb=7)
    pending.sort(key=lambda t: (t[3], -costs[t[1]], t[0], t[2]))
    started = time.perf_counter()
    status(phase="fitting", stage=stage, total=len(all_tasks), restored=restored, jobs=jobs)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(worker, task) for task in pending]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status(phase="unit_complete", stage=stage, total=len(all_tasks), done=restored+n,
                   elapsed_seconds=round(time.perf_counter()-started, 2), **result)
    immutable(OUT / f"{stage}_complete.json", dict(identity=identity, units=len(all_tasks), status="complete"))


def evaluate(identity):
    assert read(OUT / "fit_complete.json")["identity"] == identity
    if (OUT / "results_complete.json").exists():
        return
    assert verify_inputs() == identity
    centers, graphs, counts = {}, {}, {}
    for scenario in SCENARIOS:
        models = read(POINT / scenario / "prepared.json")["admitted"]
        for arm in ("plain", "augmented"):
            repeats = []
            for repeat in range(13):
                values = []
                for model in models:
                    task = (scenario, model, arm, repeat, identity)
                    assert unit_record(task) is not None
                    with np.load(profile_path(task), allow_pickle=False) as profile:
                        values.append(profile["mean"])
                repeats.append(np.median(values, axis=0))
            centers[scenario, arm] = np.array(repeats)
        for arm, gate, method in base.configurations():
            key = f"{scenario}__{arm}__{gate}__{method}"
            graphs[key], counts[key] = recurrence(centers[scenario, arm], arm, gate, method, adapter)
            np.testing.assert_array_equal(graphs[key][0], base.graphs(centers[scenario, arm][0], arm, gate, method))
            assert not (graphs[key][1:] & ~graphs[key][:-1]).any()
    np.savez_compressed(OUT / "profiles.npz", **{f"{s}__{a}": v for (s, a), v in centers.items()})
    np.savez_compressed(OUT / "graphs.npz", **graphs)
    np.savez_compressed(OUT / "counts.npz", **counts)
    immutable(OUT / "empirical_frozen.json", dict(identity=identity,
        files={name: sha(OUT / name) for name in ("profiles.npz", "graphs.npz", "counts.npz", "input_manifest.json", "fit_complete.json")}))
    refs, rows, edges, paired = {}, [], [], []
    for scenario in SCENARIOS:
        _, y, _ = sample(scenario)
        fit = read(POINT / scenario / "roles.json")["fit"]
        refs[scenario] = design.population_scores(scenario, np.bincount(y[fit], minlength=3)/len(fit))
        target = design.truth(refs[scenario])
        main_key = f"{scenario}__augmented__shadow__Joint4"
        for c, j, k in np.argwhere(graphs[main_key][0]):
            edges.append(dict(scenario=scenario, class_id=int(c), leader=FEATURES[j], follower=FEATURES[k],
                              correct=bool(target[c, j, k]), supports=int(counts[main_key][c, j, k])))
        for arm, gate, method in base.configurations():
            key = f"{scenario}__{arm}__{gate}__{method}"
            for threshold, graph in enumerate(graphs[key]):
                for name in (("common",) if method == "Joint4" else ("common", "own")):
                    own = SOURCES.index(method) if name == "own" else None
                    for c in range(3):
                        rows.append(dict(scenario=scenario, arm=arm, gate=gate, method=method, target=name,
                            threshold=threshold, class_id=c, **design.grade(graph[c], refs[scenario][:, c], own)))
                main = graphs[main_key][threshold]
                paired.append(dict(scenario=scenario, arm=arm, gate=gate, method=method, threshold=threshold,
                    shared_correct=int((main & graph & target).sum()), shared_wrong=int((main & graph & ~target).sum()),
                    main_only_correct=int((main & ~graph & target).sum()), main_only_wrong=int((main & ~graph & ~target).sum()),
                    comparator_only_correct=int((~main & graph & target).sum()), comparator_only_wrong=int((~main & graph & ~target).sum())))
    np.savez_compressed(OUT / "references.npz", **refs)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "classes.csv", index=False)
    metrics = ["correct", "wrong", "declared", "true_total", "tied_wrong", "reverse_wrong", "disagreement_wrong"]
    for name, keys in (("by_scenario", ["scenario", "arm", "gate", "method", "target", "threshold"]),
                       ("overall", ["arm", "gate", "method", "target", "threshold"])):
        ratios(frame.groupby(keys)[metrics].sum().reset_index()).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(edges).to_csv(OUT / "main_edge_support.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_comparisons.csv", index=False)
    assert verify_inputs() == identity
    immutable(OUT / "results_complete.json", dict(identity=identity, master_seed=MASTER, primary=PRIMARY,
        independent_datasets=3, scenario_draws=1, resamples_per_dataset=12, scope="fresh_confirmation_first_partition",
        graphs=len(graphs)*13, class_rows=len(frame),
        files={p.name: sha(p) for p in OUT.iterdir() if p.suffix in (".csv", ".npz")}, status="complete"))
    status(phase="evaluated", graphs=len(graphs)*13, class_rows=len(frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "prepare", "preflight", "fit", "evaluate", "all"), default="all")
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock = OUT / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        identity = contract()
        verify_contract()
        if args.stage in ("prepare", "all"):
            prepare(identity)
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

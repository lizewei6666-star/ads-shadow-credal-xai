import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import faulthandler
from pathlib import Path
import time

from . import runtime as execution


def summarize_local(args):
    from .evidence import summarize
    folder = Path(args.input).resolve()
    required = ["profiles.npz"] + (["references.npz"] if args.dataset == "simulation" else [])
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing local analysis output: {folder / name}")
    graphs, counts = folder / "graphs.npz", folder / "counts.npz"
    if graphs.exists() != counts.exists():
        raise ValueError("Decision verification requires both graphs.npz and counts.npz")
    result = summarize(folder / "profiles.npz", args.out, args.dataset,
                       references=folder / "references.npz" if args.dataset == "simulation" else None,
                       schema=execution.read(execution.ROOT / "configs/ads_schema.json"),
                       archived_graphs=graphs if graphs.exists() else None,
                       archived_counts=counts if counts.exists() else None)
    print(result, flush=True)


def run(args):
    with execution.run_lock(args.out):
        runtime, _ = execution.configure(args.dataset, args.out, args.data)
        if args.stage in ("prepare", "preflight", "all"):
            execution.prepare()
        if args.stage == "prepare":
            return
        planned = execution.tasks()
        if args.stage == "preflight":
            selected = next((t for t in planned if t[1] == "lr" and t[2] == "plain" and t[3] == "cv" and t[4] == 0), None)
            planned = [selected or planned[0]]
        check = runtime.load_unit if args.dataset == "ads" else execution.checked_unit
        pending = [task for task in planned if check(task) is None]
        completed = len(planned)-len(pending)
        started = time.monotonic()
        execution.progress("fitting", completed=completed, total=len(planned), jobs=args.jobs)
        if args.stage != "evaluate":
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
                queue = iter(pending)
                active = {}

                def submit():
                    task = next(queue, None)
                    if task is None:
                        return False
                    payload = (args.dataset, str(Path(args.out).resolve()), args.data, task)
                    active[pool.submit(execution.worker, payload)] = task
                    return True

                for _ in range(args.jobs):
                    if not submit():
                        break
                while active:
                    done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        try:
                            future.result()
                        except BaseException as exc:
                            for queued in active:
                                queued.cancel()
                            execution.progress("FAILED", task=task, error=repr(exc))
                            raise
                        completed += 1
                        submit()
                    execution.progress("fitting", completed=completed, total=len(planned),
                                       active=list(active.values()), jobs=args.jobs,
                                       elapsed_hours=(time.monotonic()-started)/3600)
        if args.stage == "preflight":
            execution.progress("PREFLIGHT_COMPLETE", completed=completed)
            return
        if any(check(task) is None for task in execution.tasks()):
            raise RuntimeError("Some units are incomplete; resume fitting before evaluation")
        if args.dataset == "ads":
            runtime.evaluate()
            from .evidence import summarize
            output = execution.STATE["out"]
            summarize(output / "profiles.npz", output / "tables", "ads",
                      schema=execution.read(execution.ROOT / "configs/ads_schema.json"))
        else:
            execution.evaluate_simulation()
        execution.progress("RUN_COMPLETE", completed=len(execution.tasks()), primary_support=8)


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Matched-shadow feature priorities")
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("summarize", help="Rebuild tables from locally held analysis outputs")
    replay.add_argument("--dataset", choices=("simulation", "ads"), required=True)
    replay.add_argument("--input", required=True)
    replay.add_argument("--out", required=True)
    replay.set_defaults(function=summarize_local)
    fitting = commands.add_parser("run", help="Fit the frozen manuscript protocol")
    fitting.add_argument("--dataset", choices=("simulation", "ads"), required=True)
    fitting.add_argument("--out", required=True)
    fitting.add_argument("--data", type=lambda p: str(Path(p).resolve()))
    fitting.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    fitting.add_argument("--stage", choices=("prepare", "preflight", "fit", "evaluate", "all"), default="all")
    fitting.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)

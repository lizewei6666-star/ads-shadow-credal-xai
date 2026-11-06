"""Reconstruct directional decisions from stored source profiles."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

SOURCES = ("SHAP", "PFI", "LOCO", "FIRM")
EPS = 1e-12


def graph_for(center, p, gate, method):
    center = np.asarray(center, float)
    if center.ndim != 5 or center.shape[1:4] != (2, 4, 3) or not np.isfinite(center).all():
        raise ValueError("Invalid source profile array")
    if p < 1 or p > center.shape[-1] or gate not in ("none", "zero", "shadow"):
        raise ValueError("Invalid feature count or qualification rule")
    scores = center[..., :p]
    direction = scores[..., :, None] - scores[..., None, :] > EPS
    if gate == "shadow":
        if center.shape[-1] != 3*p:
            raise ValueError("Each original feature requires two matched controls")
        controls = np.arange(p, 3*p).reshape(p, 2)
        qualified = (scores[..., :, None] - center[..., controls] > EPS).all(axis=-1)
    elif gate == "zero":
        qualified = scores > EPS
    else:
        qualified = np.ones_like(scores, dtype=bool)
    votes = (direction & qualified[..., :, None]).all(axis=1)
    if method == "Joint4":
        support = votes.sum(axis=1)
        passed = (support >= 2) & (support > support.swapaxes(-1, -2))
        anchor = passed[0]
        reach = anchor.copy()
        for j in range(p):
            reach |= reach[..., :, j, None] & reach[..., None, j, :]
        candidate = anchor & ~(reach & reach.swapaxes(-1, -2))
    elif method in SOURCES:
        passed = votes[:, SOURCES.index(method)]
        candidate = passed[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    counts = passed[1:].sum(axis=0)
    graphs = np.array([candidate & (counts >= k) for k in range(len(passed))])
    return graphs, counts, votes


def grade(graph, scores):
    delta = scores[..., :, None] - scores[..., None, :]
    truth = (delta > 1e-10).all(axis=0)
    correct = int((graph & truth).sum())
    wrong = int((graph & ~truth).sum())
    return dict(correct=correct, wrong=wrong, declared=correct+wrong, true_total=int(truth.sum()))


def ratios(frame):
    frame = frame.copy()
    frame["precision"] = frame.correct / frame.declared.replace(0, np.nan)
    frame["recovery"] = frame.correct / frame.true_total.replace(0, np.nan)
    return frame


def summarize(profile_path, output, domain, references=None, schema=None,
              archived_graphs=None, archived_counts=None):
    """Rebuild every comparator and tier, optionally checking archived decisions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = [f"X{i}" for i in range(1, 11)] if domain == "simulation" else schema["feature_columns"]
    p = len(features)
    old_graphs = np.load(archived_graphs, allow_pickle=False) if archived_graphs else None
    old_counts = np.load(archived_counts, allow_pickle=False) if archived_counts else None
    refs = np.load(references, allow_pickle=False) if references else None
    summaries, tiers, relations = [], [], []
    checked = 0
    try:
        with np.load(profile_path, allow_pickle=False) as profiles:
            for key in profiles.files:
                case, arm, policy, scope = key.split("__")
                center = profiles[key]
                gates = ("none", "zero") if arm == "plain" else ("none", "zero", "shadow")
                for gate in gates:
                    for method in (*SOURCES, "Joint4"):
                        graph, counts, _ = graph_for(center, p, gate, method)
                        name = f"{key}__{gate}__{method}"
                        if old_graphs is not None:
                            np.testing.assert_array_equal(graph, old_graphs[name], err_msg=name)
                            np.testing.assert_array_equal(counts, old_counts[name], err_msg=name)
                            checked += 1
                        meta = dict(case=case, arm=arm, policy=policy, scope=scope, gate=gate, method=method)
                        if domain == "simulation":
                            meta["scenario"] = case.split("-", 1)[1]
                        for c in range(3):
                            for k, g in enumerate(graph):
                                record = dict(**meta, class_id=c, support=k, declared=int(g[c].sum()))
                                if refs is not None:
                                    record.update(grade(g[c], refs[case][:, c]))
                                summaries.append(record)
                            for tier, lo, hi in (("high", 8, 12), ("medium", 4, 7), ("low", 0, 3)):
                                mask = graph[0, c] & (counts[c] >= lo) & (counts[c] <= hi)
                                record = dict(**meta, class_id=c, tier=tier, declared=int(mask.sum()))
                                if refs is not None:
                                    record.update(grade(mask, refs[case][:, c]))
                                tiers.append(record)
                            if scope == "admitted_pool" and policy == "cv" and gate == "shadow" and method == "Joint4":
                                for j, h in np.argwhere(graph[0, c]):
                                    k = int(counts[c, j, h])
                                    relations.append(dict(**meta, class_id=c, leader=features[j], follower=features[h],
                                                          support=k, tier="high" if k >= 8 else "medium" if k >= 4 else "low"))
        frame, tier_frame = pd.DataFrame(summaries), pd.DataFrame(tiers)
        frame.to_csv(output / "comparisons_by_class.csv", index=False)
        tier_frame.to_csv(output / "tiers_by_class.csv", index=False)
        pd.DataFrame(relations).to_csv(output / "candidate_relations.csv", index=False)
        if refs is not None:
            metrics = ["correct", "wrong", "declared", "true_total"]
            dims = ["arm", "policy", "scope", "gate", "method"]
            for label, table, tail in (("comparisons", frame, "support"), ("tiers", tier_frame, "tier")):
                for suffix, groups in (("pooled", []), ("by_scenario", ["scenario"])):
                    table_out = ratios(table.groupby(groups+dims+[tail], dropna=False)[metrics].sum().reset_index())
                    table_out.to_csv(output / f"{label}_{suffix}.csv", index=False)
        manifest = dict(domain=domain, profiles=str(Path(profile_path).name), decision_arrays_checked=checked,
                        primary_support=8, tiers={"high": [8, 12], "medium": [4, 7], "low": [0, 3]},
                        zero_declarations_precision=None, null_recovery=None)
        (output / "reproduction.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        return manifest
    finally:
        for value in (old_graphs, old_counts, refs):
            if value is not None:
                value.close()

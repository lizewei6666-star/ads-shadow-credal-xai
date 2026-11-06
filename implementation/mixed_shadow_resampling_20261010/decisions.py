"""Original-anchor recurrence of paired, source-specific declarations."""
import numpy as np


def recurrence(center, arm, gate, method, adapter):
    p = 10
    controls = np.arange(p, 3*p).reshape(p, 2)
    padded = np.pad(center, [(0, 0)]*4 + [(0, 2*p)]) if arm == "plain" else center
    votes = adapter.votes(padded, p, controls, gate)
    if method == "Joint4":
        result = adapter.joint_graphs(votes)
        return result["graphs"], result["counts"]
    selected = votes[:, adapter.SOURCES.index(method)]
    counts = selected[1:].sum(axis=0)
    graphs = np.array([selected[0] & (counts >= k) for k in range(len(selected))])
    return graphs, counts


def ratios(table):
    table = table.copy()
    table["precision"] = table.correct / table.declared.replace(0, np.nan)
    table["recovery"] = table.correct / table.true_total.replace(0, np.nan)
    return table

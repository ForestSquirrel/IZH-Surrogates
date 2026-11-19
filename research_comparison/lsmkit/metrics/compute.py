"""
compute.py -- compute-cost records from engine runs.
"""
import numpy as np


def merge(records):
    """sum several engine compute records (same backend/model) into one"""
    if not records:
        return {}
    out = dict(records[0])
    for k in ("wall_s", "instances", "steps", "neuron_steps", "events"):
        out[k] = sum(r[k] for r in records)
    out["neuron_steps_per_s"] = out["neuron_steps"] / out["wall_s"] if out["wall_s"] else float("nan")
    out["s_per_instance"] = out["wall_s"] / out["instances"]
    return out


def ratio(sur, tgt):
    """surrogate / target cost figures for the same workload"""
    return dict(wall=sur["wall_s"] / tgt["wall_s"],
                per_neuron_step=(sur["wall_s"] / sur["neuron_steps"]) / (tgt["wall_s"] / tgt["neuron_steps"]),
                model_substeps=sur.get("model_substeps", 1) / tgt.get("model_substeps", 1))

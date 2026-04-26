#!/usr/bin/env python3
"""
Dump configs side-by-side for Walt's reference checkpoints and the human_20k
checkpoint. Identifies which knob differs (num_layers, ffn_dim, ree_base,
feature_type, training subset size, epochs, etc.).

Run on Savio login node (no GPU needed, just torch.load + dict printing):

    python scripts/diff_walt_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

CHECKPOINTS = [
    "/home/walt/Attention/checkpoints_performer/20jo1hdd/best_model.pt",
    "/home/walt/Attention/checkpoints_performer/s66qfh36/best_model.pt",
    "checkpoints/human_20k/best_model.pt",
]


def dump(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"=== {path} ===\n  FILE NOT FOUND\n")
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"=== {path} ===\n  LOAD FAILED: {e}\n")
        return None

    print(f"=== {path} ===")
    print(f"  val_loss     = {ck.get('val_loss')}")
    print(f"  epoch        = {ck.get('epoch')}")
    print(f"  total_params = {ck.get('total_params')}")

    cfg = ck.get("config") or ck.get("cfg") or {}
    if not cfg:
        # Fall back: maybe top-level dict is itself the config
        cfg = {k: v for k, v in ck.items()
               if not hasattr(v, "shape") and k not in ("model_state_dict", "optimizer_state_dict",
                                                       "scheduler_state_dict", "state_dict",
                                                       "val_loss", "epoch", "total_params")}

    print("  --- config ---")
    for k in sorted(cfg):
        v = cfg[k]
        if hasattr(v, "shape"):
            continue
        sval = repr(v)
        if len(sval) > 120:
            sval = sval[:117] + "..."
        print(f"  {k} = {sval}")

    sd = ck.get("model_state_dict") or ck.get("state_dict") or {}
    if "gene_embedding.weight" in sd:
        print(f"  gene_embedding.weight.shape = {tuple(sd['gene_embedding.weight'].shape)}")
    n_layers_seen = sum(1 for k in sd if k.startswith("layers.") and k.endswith(".attention.W_q.weight"))
    if n_layers_seen:
        print(f"  layers seen in state_dict   = {n_layers_seen}")
    print()
    return cfg


def main():
    cfgs = {}
    for path in CHECKPOINTS:
        cfgs[path] = dump(path)

    valid = {p: c for p, c in cfgs.items() if c}
    if len(valid) < 2:
        print("Need at least 2 loadable checkpoints to diff.")
        return

    all_keys = sorted(set().union(*[set(c.keys()) for c in valid.values()]))
    print("=" * 80)
    print("DIFF (key — values across checkpoints; only keys that differ)")
    print("=" * 80)
    for k in all_keys:
        vals = {p: c.get(k, "<missing>") for p, c in valid.items()}
        unique = set(repr(v) for v in vals.values())
        if len(unique) <= 1:
            continue
        print(f"\n  [{k}]")
        for p, v in vals.items():
            print(f"    {Path(p).parent.name}/{Path(p).name}: {v!r}")


if __name__ == "__main__":
    main()

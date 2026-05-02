#!/usr/bin/env python3
"""
Training-free MoE headroom analysis on OSDR.

Loads the 3 frozen 5k experts (human, mouse, mixed), runs them on the OSDR
parquet, and computes:

    - Per-expert per-sample Pearson on masked positions (matches the
      CLAUDE.md table — sanity check).
    - Uniform 1/3 ensemble: does naive averaging help?
    - Grid-search optimal fixed weights on a 3-simplex (step 0.05).
    - Per-sample oracle: cheat by picking the best expert per sample.
      Upper bound on what ANY learned gate could achieve.

Mask is sampled in the SHARED common gene space (intersection of the 3
experts' native gene lists ∩ canonical_genes), so all experts predict at
the same positions and ensembling is well-defined. Each expert is fed in
its own native gene order (matches its training), then predictions are
gathered back to the common space.

Two outputs:
    <output-dir>/predictions.npz  — cached per-expert predictions (~hundreds
                                    of MB; analysis is then ~seconds).
    <output-dir>/report.json      — summary metrics for the presentation.

Usage:
    python analyze_moe_headroom.py \\
        --osdr-parquet data/osdr/osdr_expression.parquet \\
        --human-ckpt   checkpoints/human_5k/best_model.pt \\
        --mouse-ckpt   checkpoints/mouse_5k/best_model.pt \\
        --mixed-ckpt   checkpoints/mixed_5k/best_model.pt \\
        --output-dir   results/moe_headroom_5k

The cache step needs a GPU. Once predictions.npz exists, --analyze-only
runs entirely on CPU in seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_osdr import load_canonical_genes, load_checkpoint


EXPERT_NAMES = ("human_5k", "mouse_5k", "mixed_5k")
MASK_TOKEN = -10


# ──────────────────────────────────────────────────────────────────────────
# CACHE PHASE: run experts, save predictions
# ──────────────────────────────────────────────────────────────────────────

def _build_common_space(canonical_genes, gene_lists):
    """Returns (common_genes, common_in_canonical, common_in_native_per_expert).

    common_genes is the sorted-by-canonical-order intersection of all expert
    gene lists. common_in_canonical[j] is the index of common_genes[j] in
    canonical_genes. common_in_native_per_expert[name][j] is its index in
    that expert's native gene_list.
    """
    canonical_to_idx = {g: i for i, g in enumerate(canonical_genes)}
    common_set = set.intersection(*(set(gl) for gl in gene_lists.values()))
    # order common genes by canonical position for determinism
    common_genes = sorted(common_set, key=lambda g: canonical_to_idx[g])
    common_in_canonical = np.array([canonical_to_idx[g] for g in common_genes], dtype=np.int64)

    common_in_native = {}
    for name, gl in gene_lists.items():
        idx_map = {g: i for i, g in enumerate(gl)}
        common_in_native[name] = np.array([idx_map[g] for g in common_genes], dtype=np.int64)

    return common_genes, common_in_canonical, common_in_native


def _native_gene_idx_in_canonical(gene_list, canonical_genes):
    """For each native gene in the expert's training order, its index in canonical_genes."""
    canonical_to_idx = {g: i for i, g in enumerate(canonical_genes)}
    return np.array([canonical_to_idx[g] for g in gene_list], dtype=np.int64)


def _expert_predict_common(model, x_osdr, native_idx_in_canon, common_in_native_e,
                           mask_idx_common, batch_size, mask_token, device, name):
    """Run one expert over all OSDR samples, return predictions in common space.

    x_osdr: (N, G_canonical) float32
    native_idx_in_canon: (G_native,) — column in canonical_genes for each native position
    common_in_native_e: (G_common,) — column in native gene order for each common gene
    mask_idx_common: (N, num_mask) — masked positions per sample, in common-space indices
    """
    N = x_osdr.shape[0]
    G_native = native_idx_in_canon.shape[0]
    G_common = common_in_native_e.shape[0]

    pred_common_all = np.zeros((N, G_common), dtype=np.float32)

    # Translate common-space mask to native-space mask once.
    mask_idx_native = common_in_native_e[mask_idx_common]  # (N, num_mask)

    common_in_native_t = torch.from_numpy(common_in_native_e).to(device)
    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # Project canonical → native (slice columns).
        x_native_np = x_osdr[start:end][:, native_idx_in_canon].copy()
        # Apply mask token at native positions corresponding to the chosen common positions.
        for i in range(end - start):
            x_native_np[i, mask_idx_native[start + i]] = mask_token
        x_native = torch.from_numpy(x_native_np).to(device, non_blocking=True)
        with torch.no_grad():
            pred_native = model(x_native)  # (B, G_native)
        pred_common_batch = pred_native.index_select(1, common_in_native_t).cpu().numpy()
        pred_common_all[start:end] = pred_common_batch

        if (start // batch_size) % 25 == 0:
            elapsed = time.time() - t0
            done = end
            eta = elapsed / max(1, done) * (N - done)
            print(f"  [{name}] {done}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    print(f"  [{name}] done in {time.time() - t0:.1f}s "
          f"(G_native={G_native}, G_common={G_common})", flush=True)
    return pred_common_all


def cache_predictions(args, cache_path: Path):
    print("=" * 70)
    print("CACHE PHASE")
    print("=" * 70)

    canonical_genes = load_canonical_genes()
    print(f"[setup] canonical genes: {len(canonical_genes)}", flush=True)

    osdr_df = pd.read_parquet(args.osdr_parquet)
    missing_canon = [g for g in canonical_genes if g not in osdr_df.columns]
    if missing_canon:
        raise ValueError(
            f"OSDR parquet missing {len(missing_canon)} canonical gene columns. "
            f"Was it produced by an older preprocess? First few missing: {missing_canon[:5]}"
        )
    x_osdr = osdr_df[canonical_genes].values.astype("float32")  # (N, G_canonical)
    N = x_osdr.shape[0]
    print(f"[setup] OSDR samples: {N}, canonical-aligned shape: {x_osdr.shape}", flush=True)

    species = (osdr_df["species"].astype(str).values
               if "species" in osdr_df.columns else None)
    if species is not None:
        from collections import Counter
        print(f"[setup] species: {dict(Counter(species))}", flush=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device: {device}", flush=True)
    # SLiMPerformer OOMs on 1080Ti at ~14k genes (see scripts/savio_evaluate_osdr.sh).
    # CPU is the safe default for the cache phase on Savio.

    expert_paths = {
        "human_5k": args.human_ckpt,
        "mouse_5k": args.mouse_ckpt,
        "mixed_5k": args.mixed_ckpt,
    }
    models, gene_lists = {}, {}
    for name, p in expert_paths.items():
        print(f"[load] {name} <- {p}", flush=True)
        model, _cfg, gl = load_checkpoint(Path(p), device)
        if not gl:
            raise RuntimeError(
                f"{name}: load_checkpoint returned empty gene_list (training parquet "
                f"path stored in checkpoint config not found on disk). Cannot align "
                f"to common space without it."
            )
        models[name] = model
        gene_lists[name] = gl
        print(f"  gene_list={len(gl)}, num_genes={model.num_genes}", flush=True)

    common_genes, common_in_canonical, common_in_native = _build_common_space(
        canonical_genes, gene_lists
    )
    G_common = len(common_genes)
    print(f"\n[align] Common space: {G_common} genes "
          f"(intersection of 3 expert gene lists, canonical-ordered)", flush=True)
    for name, gl in gene_lists.items():
        print(f"  {name}: native={len(gl)}, common-coverage={G_common}/{len(gl)} "
              f"({100*G_common/len(gl):.1f}%)", flush=True)

    rng = np.random.default_rng(args.seed)
    num_mask = max(1, int(G_common * args.mask_ratio))
    mask_idx_common = np.stack(
        [rng.choice(G_common, num_mask, replace=False) for _ in range(N)]
    ).astype(np.int64)
    print(f"\n[mask] random_{args.mask_ratio:.0%} mask: {num_mask} positions/sample, "
          f"shape {mask_idx_common.shape}", flush=True)

    gt_common = x_osdr[:, common_in_canonical]  # (N, G_common)

    print("\n[predict] Running each expert...", flush=True)
    preds = {}
    for name, model in models.items():
        native_idx = _native_gene_idx_in_canonical(gene_lists[name], canonical_genes)
        preds[name] = _expert_predict_common(
            model=model,
            x_osdr=x_osdr,
            native_idx_in_canon=native_idx,
            common_in_native_e=common_in_native[name],
            mask_idx_common=mask_idx_common,
            batch_size=args.batch_size,
            mask_token=MASK_TOKEN,
            device=device,
            name=name,
        )
        # Free the model — predictions are on CPU now.
        del model
        models[name] = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[save] {cache_path}", flush=True)
    save_kwargs = dict(
        common_genes=np.array(common_genes, dtype=object),
        common_in_canonical=common_in_canonical,
        mask_idx_common=mask_idx_common,
        gt_common=gt_common,
        pred_human=preds["human_5k"],
        pred_mouse=preds["mouse_5k"],
        pred_mixed=preds["mixed_5k"],
        mask_ratio=np.float32(args.mask_ratio),
        seed=np.int64(args.seed),
    )
    if species is not None:
        save_kwargs["species"] = np.array(species, dtype=object)
    np.savez_compressed(cache_path, **save_kwargs)
    sz_mb = cache_path.stat().st_size / (1024 ** 2)
    print(f"[save] done ({sz_mb:.1f} MiB)", flush=True)


# ──────────────────────────────────────────────────────────────────────────
# ANALYSIS PHASE: load cache, compute headroom metrics
# ──────────────────────────────────────────────────────────────────────────

def per_sample_pearson(pred, gt, mask_idx):
    """pred, gt: (N, G). mask_idx: (N, K) or list of length-N arrays. Returns (N,)."""
    rs = np.empty(len(pred), dtype=np.float64)
    for i in range(len(pred)):
        idx = mask_idx[i]
        if len(idx) < 3:
            rs[i] = np.nan; continue
        p = pred[i, idx]; t = gt[i, idx]
        if np.std(p) < 1e-9 or np.std(t) < 1e-9:
            rs[i] = np.nan; continue
        rs[i] = np.corrcoef(p, t)[0, 1]
    return rs


def gene_mean_baseline_pearson(gt, mask_idx):
    """Predict per-gene mean (computed across the OSDR samples) at each masked position."""
    gene_mean = gt.mean(axis=0)  # (G,)
    pred = np.broadcast_to(gene_mean, gt.shape)
    return per_sample_pearson(pred, gt, mask_idx)


def grid_search_weights(preds, gt, mask_idx, step=0.05):
    """Search the 3-simplex; pick weights maximizing mean per-sample pearson.
    preds: (E, N, G). Returns (best_weights, best_mean, sorted_top10)."""
    levels = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best_w, best_score = None, -np.inf
    results = []
    for wh in levels:
        for wm in levels:
            wx = round(1.0 - wh - wm, 4)
            if wx < -1e-9 or wx > 1.0 + 1e-9: continue
            wx = max(0.0, min(1.0, wx))
            pred = wh * preds[0] + wm * preds[1] + wx * preds[2]
            r = per_sample_pearson(pred, gt, mask_idx)
            score = float(np.nanmean(r))
            results.append({"w_human": float(wh), "w_mouse": float(wm),
                            "w_mixed": float(wx), "mean_pearson": score})
            if score > best_score:
                best_score = score; best_w = (float(wh), float(wm), float(wx))
    results.sort(key=lambda r: -r["mean_pearson"])
    return best_w, best_score, results[:10]


def oracle_per_sample(per_expert_r):
    """per_expert_r: (E, N). Returns max along axis 0."""
    return np.nanmax(per_expert_r, axis=0)


def _summarize(name, r):
    return {
        "name": name,
        "mean": float(np.nanmean(r)),
        "median": float(np.nanmedian(r)),
        "std": float(np.nanstd(r)),
        "n_valid": int(np.sum(~np.isnan(r))),
    }


def _filter_mask_to_topvar(mask_idx, topvar_set, min_kept=3):
    """Per sample, keep mask positions that fall in topvar_set. Returns list of arrays.
    Samples with fewer than min_kept retained positions get an empty array (→ NaN pearson)."""
    out = []
    for i in range(len(mask_idx)):
        kept = mask_idx[i][np.isin(mask_idx[i], topvar_set)]
        out.append(kept if len(kept) >= min_kept else np.array([], dtype=np.int64))
    return out


def analyze(args, cache_path: Path, out_dir: Path):
    print("=" * 70)
    print("ANALYSIS PHASE")
    print("=" * 70)
    z = np.load(cache_path, allow_pickle=True)
    common_genes = list(z["common_genes"])
    mask_idx = z["mask_idx_common"]
    gt = z["gt_common"]
    pred_h = z["pred_human"]; pred_m = z["pred_mouse"]; pred_x = z["pred_mixed"]
    mask_ratio = float(z["mask_ratio"])
    seed = int(z["seed"])
    species = z["species"] if "species" in z.files else None
    N, G_common = gt.shape
    print(f"[load] N={N}  G_common={G_common}  mask_ratio={mask_ratio}  seed={seed}", flush=True)
    if species is not None:
        from collections import Counter
        print(f"[load] species: {dict(Counter(species))}", flush=True)

    preds_arr = np.stack([pred_h, pred_m, pred_x], axis=0)  # (3, N, G_common)

    def report_block(label, mask):
        print(f"\n--- {label} ---")
        rows = []
        # singles
        per_expert_r = []
        for name, p in zip(EXPERT_NAMES, preds_arr):
            r = per_sample_pearson(p, gt, mask)
            per_expert_r.append(r)
            rows.append(_summarize(name, r))
        per_expert_r = np.stack(per_expert_r, axis=0)  # (3, N)

        n_valid_singles = max(row["n_valid"] for row in rows)
        if n_valid_singles == 0:
            n_kept = sum(len(m) for m in mask) if isinstance(mask, list) else int((mask >= 0).sum())
            print(f"  [skip] no sample has a valid per-sample Pearson "
                  f"(retained mask positions across all samples = {n_kept}). "
                  f"Block produced nothing comparable; emitting empty rows.")
            return rows, []

        # uniform
        r_uni = per_sample_pearson((pred_h + pred_m + pred_x) / 3.0, gt, mask)
        rows.append(_summarize("uniform_1/3", r_uni))

        # grid-best
        best_w, best_score, top10 = grid_search_weights(preds_arr, gt, mask)
        if best_w is None:
            rows.append({"name": "grid_best (no valid score)", "mean": float("nan"),
                         "median": float("nan"), "std": float("nan"), "n_valid": 0,
                         "weights": None})
        else:
            wh, wm, wx = best_w
            grid_pred = wh * pred_h + wm * pred_m + wx * pred_x
            r_grid = per_sample_pearson(grid_pred, gt, mask)
            rows.append({**_summarize(f"grid_best (h={wh:.2f},m={wm:.2f},x={wx:.2f})", r_grid),
                         "weights": [wh, wm, wx]})

        # oracle (per-sample best expert)
        r_oracle = oracle_per_sample(per_expert_r)
        rows.append(_summarize("oracle (per-sample)", r_oracle))

        # gene_mean baseline
        r_gm = gene_mean_baseline_pearson(gt, mask)
        rows.append(_summarize("baseline_gene_mean", r_gm))

        # print
        print(f"  {'condition':<35} {'mean':>8}  {'median':>8}  {'std':>6}  {'n':>5}")
        for row in rows:
            print(f"  {row['name']:<35} {row['mean']:>+8.4f}  {row['median']:>+8.4f}  "
                  f"{row['std']:>6.4f}  {row['n_valid']:>5d}")
        # gaps
        best_single = max(rows[i]["mean"] for i in range(3))
        oracle = rows[5]["mean"]
        uni = rows[3]["mean"]
        grid = rows[4]["mean"]
        print(f"\n  gap (oracle - best_single)   = {oracle - best_single:+.4f}   ← MoE headroom")
        print(f"  gap (grid   - best_single)   = {grid   - best_single:+.4f}   ← fixed-weight win")
        print(f"  gap (uni    - best_single)   = {uni    - best_single:+.4f}   ← naive ensemble win")
        return rows, top10

    # full common-gene mask
    rows_all, top10_all = report_block(f"All common genes (mask {mask_ratio:.0%})",
                                       mask_idx)

    # high-variance subset (top-K by per-gene variance over OSDR samples)
    gene_var = gt.var(axis=0)
    topk = min(args.high_var_topk, G_common)
    top_idx = np.argsort(gene_var)[-topk:]
    top_set = set(top_idx.tolist())
    mask_topvar = _filter_mask_to_topvar(mask_idx, top_set)
    rows_hv, top10_hv = report_block(
        f"High-variance subset (top {topk} of {G_common} common genes by OSDR variance)",
        mask_topvar,
    )

    # per-species breakdown (only meaningful when the eval set has >1 species).
    # The MoE headroom test needs samples that prefer different experts; with a
    # single-species set, oracle ≈ best_single by construction.
    species_breakdown = None
    if species is not None and len(set(species)) > 1:
        print("\n--- By species (mask {:.0%}) ---".format(mask_ratio))
        # recompute per-expert per-sample r once on the all-common mask
        per_expert_r = np.stack([
            per_sample_pearson(p, gt, mask_idx)
            for p in (pred_h, pred_m, pred_x)
        ], axis=0)  # (3, N)
        r_oracle = oracle_per_sample(per_expert_r)
        r_uni = per_sample_pearson((pred_h + pred_m + pred_x) / 3.0, gt, mask_idx)

        species_breakdown = {}
        header = (f"  {'species':<8} {'N':>5}  "
                  + "  ".join(f"{n:>9}" for n in EXPERT_NAMES)
                  + f"  {'uniform':>9}  {'oracle':>9}  best→")
        print(header)
        for sp in sorted(set(species)):
            mask = species == sp
            nsp = int(mask.sum())
            per_expert_means = [float(np.nanmean(per_expert_r[i, mask])) for i in range(3)]
            uni_mean = float(np.nanmean(r_uni[mask]))
            ora_mean = float(np.nanmean(r_oracle[mask]))
            best_idx = int(np.argmax(per_expert_means))
            best_name = EXPERT_NAMES[best_idx]
            print(f"  {sp:<8} {nsp:>5}  "
                  + "  ".join(f"{v:>+9.4f}" for v in per_expert_means)
                  + f"  {uni_mean:>+9.4f}  {ora_mean:>+9.4f}  {best_name}")
            species_breakdown[sp] = {
                "n": nsp,
                "per_expert_mean": dict(zip(EXPERT_NAMES, per_expert_means)),
                "uniform_mean": uni_mean,
                "oracle_mean": ora_mean,
                "best_single": best_name,
                "gap_oracle_minus_best": ora_mean - max(per_expert_means),
            }
        # global oracle (per-sample best across the FULL mixed set), for reference
        r_oracle_all = float(np.nanmean(r_oracle))
        r_best_all = max(float(np.nanmean(per_expert_r[i])) for i in range(3))
        print(f"\n  global oracle (mixed)    = {r_oracle_all:+.4f}")
        print(f"  global best-single       = {r_best_all:+.4f}")
        print(f"  gap (oracle - best_single) = {r_oracle_all - r_best_all:+.4f}   "
              f"← MoE headroom on MIXED eval")
        species_breakdown["__global__"] = {
            "oracle_mean": r_oracle_all,
            "best_single_mean": r_best_all,
            "gap_oracle_minus_best": r_oracle_all - r_best_all,
        }

    report = {
        "n_samples": N,
        "n_common_genes": G_common,
        "mask_ratio": mask_ratio,
        "seed": seed,
        "all_common": rows_all,
        "high_variance": {"topk": topk, "rows": rows_hv},
        "grid_top10_all_common": top10_all,
        "grid_top10_high_variance": top10_hv,
        "by_species": species_breakdown,
    }
    out_path = out_dir / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdr-parquet", required=True)
    ap.add_argument("--human-ckpt", required=True)
    ap.add_argument("--mouse-ckpt", required=True)
    ap.add_argument("--mixed-ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="CPU-friendly default; lower for GPU if OOM.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto",
                    help="Auto picks CUDA if available; CPU is safer for SLiMPerformer at 14k genes.")
    ap.add_argument("--high-var-topk", type=int, default=1000)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "predictions.npz"

    if args.analyze_only and not cache_path.exists():
        raise FileNotFoundError(f"--analyze-only set but {cache_path} doesn't exist")

    if not args.analyze_only:
        cache_predictions(args, cache_path)
    if not args.cache_only:
        analyze(args, cache_path, out_dir)


if __name__ == "__main__":
    main()

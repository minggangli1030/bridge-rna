"""
Distributed Data Parallel (DDP) Training for SimplePerformer
Uses both 3090 GPUs with torch.distributed
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib not installed")

# Import model from train.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import SimplePerformer, RotaryExpressionEmbedding

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'dim': 256,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 5,
    'patience': 5,
    'data_dir': './data/archs4/train_orthologs',
    'checkpoint_dir': './checkpoints_performer',
}


# ============================================================
# DATASET WITH MLM-STYLE MASKING
# ============================================================
class ExpressionMLMDataset(Dataset):
    """
    Expression dataset with masked language modeling (MLM) support.
    Randomly masks some genes' expression values and trains to reconstruct them.
    """
    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10):
        """
        Args:
            expr_array: [samples, genes] numpy array
            mask_ratio: Fraction of genes to mask per sample
            mask_token: Value to use for masked positions
        """
        self.X = expr_array.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]
        
        # Randomly mask genes
        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)
        
        x_masked = x.copy()
        x_masked[mask_indices] = self.mask_token
        
        return (
            torch.tensor(x_masked, dtype=torch.float32),      # Masked input
            torch.tensor(x, dtype=torch.float32),             # Ground truth
            torch.tensor(mask_indices, dtype=torch.long),     # Mask indices
        )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    script_start = time.time()
    
    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    is_main = rank == 0
    
    if is_main:
        print("\n" + "="*70)
        print(f"SimplePerformer Training with DDP ({world_size} GPUs)")
        print("="*70)
        print(f"\n[SETUP] Rank: {rank}, Device: {device}, World size: {world_size}")
        print("="*70)
        print("\n[VALIDATION] Data shape expectations:")
        print(f"  Expected genes: ~16,109")
        print(f"  Expected train samples: 10,000")
        print(f"  Expected val samples: 1,000")
        print(f"  (Subset for test run)")
        print("="*70)
    
    # ─────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[DATA] Loading training data...")
    
    t_start = time.time()
    
    # Load human and mouse data
    data_dir = Path(CONFIG['data_dir'])
    train_human_path = data_dir / 'train' / 'expression_train_human.parquet'
    train_mouse_path = data_dir / 'train' / 'expression_train_mouse.parquet'
    
    X_train_list = []
    
    if train_human_path.exists():
        X_human = pd.read_parquet(train_human_path).values.astype(np.float32)[:5000, :]  # Subset immediately
        X_train_list.append(X_human)
        if is_main:
            print(f"  ✓ Human train: {X_human.shape}")
    
    if train_mouse_path.exists():
        X_mouse = pd.read_parquet(train_mouse_path).values.astype(np.float32)[:5000, :]  # Subset immediately
        X_train_list.append(X_mouse)
        if is_main:
            print(f"  ✓ Mouse train: {X_mouse.shape}")
    
    X_train = np.vstack(X_train_list) if len(X_train_list) > 1 else X_train_list[0]
    num_samples, num_genes = X_train.shape
    
    t_load = time.time() - t_start
    if is_main:
        print(f"  ✓ Combined train shape: {X_train.shape}, Time: {t_load:.2f}s")
    
    # Load validation data (load on all ranks - it's only 1000 samples, not expensive)
    if is_main:
        print("\n[DATA] Loading validation data...")
    
    t_start = time.time()
    val_human_path = data_dir / 'val' / 'expression_val_human.parquet'
    val_mouse_path = data_dir / 'val' / 'expression_val_mouse.parquet'
    
    X_val_list = []
    
    if val_human_path.exists():
        X_val_h = pd.read_parquet(val_human_path).values.astype(np.float32)[:1000, :]  # Subset immediately
        X_val_list.append(X_val_h)
    
    if val_mouse_path.exists():
        X_val_m = pd.read_parquet(val_mouse_path).values.astype(np.float32)[:1000, :]  # Subset immediately
        X_val_list.append(X_val_m)
    
    X_val = np.vstack(X_val_list) if len(X_val_list) > 1 else X_val_list[0]
    t_load = time.time() - t_start
    
    if is_main:
        print(f"  ✓ Validation shape: {X_val.shape}, Time: {t_load:.2f}s")
    
    # ─────────────────────────────────────────────────────────
    # CREATE DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[DATA] Creating datasets and dataloaders...")
    
    t_start = time.time()
    
    train_dataset = ExpressionMLMDataset(X_train, mask_ratio=0.15)
    val_dataset = ExpressionMLMDataset(X_val, mask_ratio=0.15)
    
    # DistributedSampler for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    t_prep = time.time() - t_start
    if is_main:
        print(f"  ✓ Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
        print(f"  ✓ Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
        print(f"  ✓ Batch size: {CONFIG['batch_size']}, Time: {t_prep:.2f}s")
    
    # ─────────────────────────────────────────────────────────
    # BUILD MODEL
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[MODEL] Initializing SimplePerformer...")
    
    t_start = time.time()
    
    model = SimplePerformer(
        num_genes=num_genes,
        dim=CONFIG['dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True
    )
    
    t_init = time.time() - t_start
    total_params = sum(p.numel() for p in model.parameters())
    
    if is_main:
        print(f"  ✓ Parameters: {total_params:,}")
        print(f"  ✓ Init time: {t_init:.2f}s")
    
    # ─────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[OPTIM] Setting up optimizer...")
    
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs']
    )
    
    if is_main:
        print(f"  ✓ AdamW optimizer ready (lr={CONFIG['learning_rate']})")
    
    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[TRAIN] Starting training...")
        print("="*70 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create checkpoint directory
    ckpt_dir = Path(CONFIG['checkpoint_dir'])
    if is_main:
        ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
        # Set epoch for sampler (important for DDP)
        train_sampler.set_epoch(epoch)
        
        # ─────────────────────────────────────────────────────
        # TRAINING
        # ─────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(train_loader):
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)
            
            # Forward pass
            pred = model(x_masked)  # [B, G]
            
            # Loss: only on masked positions
            loss_list = []
            for i in range(len(x_masked)):
                idxs = mask_idx[i]
                if len(idxs) > 0:
                    loss_list.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))
            
            if loss_list:
                loss = torch.stack(loss_list).mean()
            else:
                loss = torch.tensor(0.0, device=device)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Heartbeat every 60 seconds (rank 0 only)
            if is_main:
                now = time.time()
                if not hasattr(main, "_last_beat") or now - main._last_beat > 60:
                    main._last_beat = now
                    pct = 100.0 * batch_idx / len(train_loader)
                    elapsed = (now - script_start) / 60
                    print(f"[HEARTBEAT] Epoch {epoch+1}/{CONFIG['epochs']} | "
                          f"Batch {batch_idx}/{len(train_loader)} ({pct:.1f}%) | "
                          f"Elapsed: {elapsed:.1f} min")
            
            # Progress every 25% of batches
            if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg_loss = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.6f} | Avg: {avg_loss:.6f}")
        
        epoch_time = time.time() - epoch_start
        epoch_train_loss = running_loss / max(1, num_batches)
        
        # ─────────────────────────────────────────────────────
        # VALIDATION
        # ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_masked, x_true, mask_idx in val_loader:
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                
                pred = model(x_masked)
                
                loss_list = []
                for i in range(len(x_masked)):
                    idxs = mask_idx[i]
                    if len(idxs) > 0:
                        loss_list.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))
                
                if loss_list:
                    val_loss += torch.stack(loss_list).mean().item()
                    val_batches += 1
        
        # Synchronize validation metrics across all ranks
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_batches_tensor = torch.tensor(val_batches, dtype=torch.float32, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = (val_loss_tensor / max(1.0, val_batches_tensor)).item()
        
        # Track losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Scheduler step
        scheduler.step()
        
        # ─────────────────────────────────────────────────────
        # CHECKPOINTING & EARLY STOPPING
        # ─────────────────────────────────────────────────────
        if is_main:
            model_to_save = model.module if isinstance(model, DDP) else model
            
            print(f"\n  ╔════════════════════════════════════════════╗")
            print(f"  ║ Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  ║ Train Loss: {epoch_train_loss:.6f}")
            print(f"  ║ Val Loss:   {epoch_val_loss:.6f}")
            print(f"  ║ Time: {epoch_time:.2f}s")
            
            # Save checkpoint for this epoch
            pt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
            torch.save(model_to_save.state_dict(), pt_path)
            
            # Check for improvement
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                # Save best model
                best_path = ckpt_dir / "best_model.pt"
                torch.save(model_to_save.state_dict(), best_path)
                print(f"  ║ ✓ Val loss improved! Saved → {best_path}")
            else:
                patience_counter += 1
                print(f"  ║ ✗ No improvement ({patience_counter}/{CONFIG['patience']})")
                
                if patience_counter >= CONFIG['patience']:
                    print(f"  ║ ⚠ Early stopping triggered!")
                    print(f"  ╚════════════════════════════════════════════╝\n")
                    break
            
            print(f"  ╚════════════════════════════════════════════╝\n")
    
    # ─────────────────────────────────────────────────────────
    # SAVE FINAL ARTIFACTS
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[SAVE] Saving final model artifacts...")
        save_start = time.time()
        
        model_to_save = model.module if isinstance(model, DDP) else model
        
        # Config JSON
        config = {
            "model_type": "simple_performer",
            "num_genes": num_genes,
            "dim": CONFIG['dim'],
            "num_heads": CONFIG['num_heads'],
            "num_layers": CONFIG['num_layers'],
            "dropout": CONFIG['dropout'],
            "final_epoch": epoch,
            "best_val_loss": best_val_loss,
            "early_stopped": patience_counter >= CONFIG['patience'],
        }
        config_path = ckpt_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Config: {config_path}")
        
        # Loss history CSV
        loss_df = pd.DataFrame({
            "epoch": range(len(train_losses)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        })
        loss_csv = ckpt_dir / "loss_history.csv"
        loss_df.to_csv(loss_csv, index=False)
        print(f"  ✓ Loss history: {loss_csv}")
        
        # Plot loss curves
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker='o', label='Train Loss', linewidth=2)
            plt.plot(val_losses, marker='s', label='Val Loss', linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("SimplePerformer Training Progress", fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = ckpt_dir / "loss_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Loss plot: {plot_path}")
        
        save_time = time.time() - save_start
        print(f"\n  ✓ Checkpoints in {ckpt_dir}/, Time: {save_time:.2f}s")
        
        # Summary
        total_time = time.time() - script_start
        print("\n" + "="*70)
        print(f"Training completed!")
        print(f"  Total script time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Final epoch: {epoch+1}/{CONFIG['epochs']}")
        print("="*70 + "\n")
    
    # Cleanup DDP
    dist.destroy_process_group()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()

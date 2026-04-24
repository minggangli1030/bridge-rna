"""
Simple Performer Model with REE and Sample-Level Embeddings
Trains on expression data: samples × genes
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from performer_pytorch import Performer

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'dim': 256,               # Embedding dimension
    'num_heads': 8,           # Attention heads
    'num_layers': 4,          # Performer blocks
    'dropout': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/archs4/train_orthologs',
}

print(f"Device: {CONFIG['device']}")


# ============================================================
# ROTARY EXPRESSION EMBEDDING (REE)
# ============================================================
class RotaryExpressionEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE): Converts continuous gene expression 
    values into sinusoidal rotation features.
    
    Modulates rotary positional encodings using expression magnitude.
    Includes masking support for special tokens (e.g., masked expression = -10).
    """
    def __init__(self, dim, base=100.0, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.mask_token_id = mask_token_id
        
        # inv_freq for sinusoidal encoding
        # Even dimensions because rotary embeddings use pairs (sin, cos)
        # base=100 (from original code) vs 10000 (standard Transformer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Use nn.Parameter to allow moving to device with model
        # requires_grad=False means it's not trained
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values
        
        Returns:
            [batch_size, num_genes, dim] sinusoidal encodings
        """
        # Identify masked positions (if any)
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)
        
        # Multiply expression values by frequencies: [B, G] x [D/2] → [B, G, D/2]
        # einsum "bi,j->bij" means: batch_idx, gene_idx paired with freq_idx
        freqs = torch.einsum("bi,j->bij", x, self.inv_freq)
        
        # Apply sin and cos, then concatenate: [B, G, D/2] → [B, G, D]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        
        # Mask out special token positions (set to 0)
        if len(x_mask_idx) > 0:
            emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0
        
        return emb


# ============================================================
# EXPRESSION DATASET
# ============================================================
class ExpressionDataset(Dataset):
    """Load expression data from parquet files.
    
    Expects: samples × genes format
    """
    def __init__(self, parquet_path, meta_path=None, standardize=False):
        """
        Args:
            parquet_path: Path to expression parquet file
            meta_path: Optional metadata CSV
            standardize: Apply z-score normalization per sample (default: False)
                        Data is already log1p TPM normalized from preprocessing
        """
        print(f"\n📂 Loading expression data from {parquet_path}...")
        
        # Load expression: should be [samples, genes]
        self.expr = pd.read_parquet(parquet_path).values.astype(np.float32)
        
        print(f"   Shape: {self.expr.shape} (samples × genes)")
        
        # Validate: samples should be >> genes (usually 100k samples, 16k genes)
        if self.expr.shape[0] < self.expr.shape[1]:
            raise ValueError(
                f"⚠️  Expected samples × genes, got {self.expr.shape}. "
                f"Transposing may be needed."
            )
        
        # Optional: z-score normalization per sample
        # Data is already log1p(TPM) normalized - only use this if needed for stability
        self.standardize = standardize
        if standardize:
            print("   Applying z-score normalization per sample...")
            mean = self.expr.mean(axis=1, keepdims=True)
            std = self.expr.std(axis=1, keepdims=True) + 1e-8
            self.expr = (self.expr - mean) / std
            self.expr = np.nan_to_num(self.expr, nan=0.0)
        
        # Load metadata if provided
        self.metadata = None
        if meta_path and os.path.exists(meta_path):
            self.metadata = pd.read_csv(meta_path)
            print(f"   Metadata: {len(self.metadata)} samples")
        
        print(f"   ✅ Loaded {self.expr.shape[0]:,} samples × {self.expr.shape[1]:,} genes")

    def __len__(self):
        return self.expr.shape[0]

    def __getitem__(self, idx):
        """Return one sample: [num_genes] expression vector"""
        return torch.from_numpy(self.expr[idx])


# ============================================================
# SIMPLE PERFORMER MODEL (No GCN)
# ============================================================
class SimplePerformer(nn.Module):
    """
    Lightweight Performer model with:
    - Gene Token Embedding (learned, per-gene identity)
    - Rotary Expression Embedding (REE, non-learnable, expression-based)
    - Sample-level Embedding (learned, global sample context)
    - Multiple Performer blocks for attention over genes
    
    Input: [batch_size, num_genes] expression
    Output: [batch_size, num_genes] reconstructed expression
    """
    def __init__(self, num_genes, dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.num_genes = num_genes
        self.dim = dim
        
        # 1. Gene Token Embedding (learned embedding per gene)
        # Similar to BERT's token embeddings - one vector per gene identity
        # Shape: [num_genes, dim]
        self.gene_token_emb = nn.Embedding(num_genes, dim)
        
        # 2. Rotary Expression Embedding (REE) - non-learnable
        # Converts continuous expression values to sinusoidal features
        self.ree = RotaryExpressionEmbedding(dim)
        
        # 3. Sample-level embedding layer
        # Learns a global context vector per sample
        # Input: [B, G] → compress to sample-level latent → expand to [B, 1, D]
        self.sample_emb = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
        
        # 3. Projection to merge REE + sample embedding
        self.proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        
        # 4. Stack of Performer blocks
        # Each block does multi-head attention over genes
        self.performers = nn.ModuleList([
            Performer(
                dim=dim,
                heads=num_heads,
                depth=1,
                dim_head=dim // num_heads,
                attn_dropout=dropout,
                ff_dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 5. LayerNorm for stability
        self.norm = nn.LayerNorm(dim)
        
        # 6. Prediction head: gene embedding → expression value
        self.head = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values
        
        Returns:
            [batch_size, num_genes] predicted expression
        """
        B, G = x.shape
        
        # Create gene indices for embedding lookup: [num_genes]
        gene_indices = torch.arange(G, device=x.device)
        
        # 1. Gene Token Embedding: [G] → [G, D] → broadcast to [B, G, D]
        gene_token_emb = self.gene_token_emb(gene_indices).unsqueeze(0)  # [1, G, D]
        
        # 2. Rotary Expression Embedding: [B, G] → [B, G, D]
        expr_emb = self.ree(x)
        
        # 3. Sample-level embedding: [B, G] → [B, D] → [B, 1, D]
        sample_emb = self.sample_emb(x).unsqueeze(1)  # [B, 1, D]
        
        # 4. Combine all three: [B, G, D] + [B, G, D] + [B, 1, D]
        #    Gene tokens identify each gene
        #    Expression embedding modulates based on value (REE)
        #    Sample embedding provides global context
        x_emb = gene_token_emb + expr_emb + sample_emb
        
        # 4. Project and mix
        x_emb = self.proj(x_emb)
        
        # 5. Apply Performer blocks (multi-head attention over genes)
        for performer in self.performers:
            x_emb = performer(x_emb) + x_emb  # Residual connection
        
        # 6. Normalize
        x_emb = self.norm(x_emb)
        
        # 7. Predict: [B, G, D] → [B, G, 1] → [B, G]
        pred = self.head(x_emb).squeeze(-1)
        
        # Ensure non-negative (expression can't be negative)
        pred = F.relu(pred)
        
        return pred


# ============================================================
# TRAINING
# ============================================================
def load_data(data_dir, split='train', batch_size=32):
    """Load expression data and create DataLoader."""
    
    split_dir = Path(data_dir) / split
    
    # Try to find parquet files
    expr_human = split_dir / f"expression_{split}_human.parquet"
    expr_mouse = split_dir / f"expression_{split}_mouse.parquet"
    meta_file = split_dir / f"metadata_{split}.csv"
    
    # Load both species if available
    datasets = []
    
    if expr_human.exists():
        print(f"\n✅ Loading HUMAN {split} data...")
        ds = ExpressionDataset(str(expr_human), str(meta_file) if meta_file.exists() else None)
        datasets.append(ds)
    
    if expr_mouse.exists():
        print(f"✅ Loading MOUSE {split} data...")
        ds = ExpressionDataset(str(expr_mouse), str(meta_file) if meta_file.exists() else None)
        datasets.append(ds)
    
    if not datasets:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")
    
    # Combine datasets
    if len(datasets) > 1:
        # Create combined dataset
        combined_expr = np.vstack([ds.expr for ds in datasets])
        
        class CombinedDataset(Dataset):
            def __init__(self, expr):
                self.expr = expr
            def __len__(self):
                return len(self.expr)
            def __getitem__(self, idx):
                return torch.from_numpy(self.expr[idx])
        
        dataset = CombinedDataset(combined_expr)
        print(f"   Combined: {len(dataset):,} samples × {dataset.expr.shape[1]:,} genes")
    else:
        dataset = datasets[0]
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    return loader, dataset.expr.shape[1]


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, x in enumerate(tqdm(loader, desc="Training")):
        x = x.to(device)
        
        # Forward pass
        pred = model(x)
        
        # Loss: MSE between predicted and original expression
        loss = F.mse_loss(pred, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    
    for x in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, x)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def main():
    print("\n" + "="*70)
    print("TRAINING: Simple Performer with REE and Sample Embeddings")
    print("="*70)
    
    # Load data
    print("\n📊 Loading training data...")
    train_loader, num_genes = load_data(
        CONFIG['data_dir'],
        split='train',
        batch_size=CONFIG['batch_size']
    )
    
    print("\n📊 Loading validation data...")
    val_loader, _ = load_data(
        CONFIG['data_dir'],
        split='val',
        batch_size=CONFIG['batch_size']
    )
    
    print(f"\n✅ Data loaded: {num_genes:,} genes per sample")
    
    # Create model
    print("\n🧠 Creating model...")
    model = SimplePerformer(
        num_genes=num_genes,
        dim=CONFIG['dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 Starting training...")
    print("="*70)
    
    best_val_loss = float('inf')
    os.makedirs('./models', exist_ok=True)
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'─'*70}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'])
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = eval_epoch(model, val_loader, CONFIG['device'])
        print(f"Val Loss:   {val_loss:.6f}")
        
        # Schedule
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                './models/best_performer.pt'
            )
            print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to ./models/best_performer.pt")


if __name__ == "__main__":
    main()

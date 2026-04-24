"""
Expression-based Performer Model (adapted from Google's SLiMPerformer)
- Simplified for continuous expression data (not discrete tokens)
- Uses Rotary Expression Embedding (REE) for positional encoding
- MLM-style training: mask genes and reconstruct expression
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryExpressionEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE): Converts continuous gene expression 
    values into sinusoidal rotation features.
    
    Modulates rotary positional encodings using expression magnitude.
    Includes masking support for special tokens (e.g., masked expression = -10).
    Original base=100 (from Google SLiMPerformer research).
    """
    def __init__(self, dim, base=100.0, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.mask_token_id = mask_token_id
        
        # inv_freq for sinusoidal encoding
        # base=100 (from original SLiMPerformer code) vs 10000 (standard Transformer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values
        
        Returns:
            [batch_size, num_genes, dim] sinusoidal encodings
        """
        # Identify masked positions
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)
        
        # Multiply expression values by frequencies: [B, G] x [D/2] → [B, G, D/2]
        freqs = torch.einsum("bi,j->bij", x, self.inv_freq)
        
        # Apply sin and cos, then concatenate: [B, G, D/2] → [B, G, D]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        
        # Mask out special token positions (set to 0)
        if len(x_mask_idx) > 0:
            emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0
        
        return emb


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism (simplified from Google SLiMPerformer)."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_genes, hidden_dim]
        Returns:
            [batch, num_genes, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        Q = self.query(x)  # [batch, seq_len, hidden_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # [batch, num_heads, seq_len, seq_len]
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        context = context.view(batch_size, seq_len, hidden_dim)  # [batch, seq_len, hidden_dim]
        
        # Final linear projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output


class FeedForward(nn.Module):
    """FFN block: linear -> gelu -> linear (from Google SLiMPerformer)."""
    
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    """Single transformer layer: attention + FFN with residual connections and layer norm."""
    
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Attention block with residual connection
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # FFN block with residual connection
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x


class ExpressionPerformer(nn.Module):
    """
    Expression Performer: Simplified Performer for continuous expression data.
    - Input: [batch, num_genes] expression values
    - Output: [batch, num_genes] predicted expression values
    - Architecture: REE embedding -> N transformer blocks -> output projection
    """
    
    def __init__(self, num_genes, hidden_dim=256, num_heads=8, num_layers=4, 
                 ffn_dim=1024, dropout=0.1, ree_base=100):
        super().__init__()
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        
        # Gene identity embedding (learnable, like BERT's token embedding)
        # Maps each gene ID to a hidden_dim vector
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)
        
        # Rotary Expression Embedding (REE) for positional encoding based on values
        self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base)
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection: hidden_dim -> 1 (reconstruct single expression value per gene)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, expr_values, mask_token_id=-10):
        """
        Args:
            expr_values: [batch, num_genes] expression values (may include masked positions)
            mask_token_id: value indicating masked positions (default: -10)
        Returns:
            predictions: [batch, num_genes] predicted expression values
        """
        batch_size, num_genes = expr_values.shape
        device = expr_values.device
        
        # Get gene identity embeddings: [num_genes, hidden_dim]
        gene_ids = torch.arange(num_genes, device=device)
        gene_emb = self.gene_embedding(gene_ids)  # [num_genes, hidden_dim]
        
        # Get REE embeddings: [batch, num_genes, hidden_dim]
        ree_emb = self.ree(expr_values, mask_token_id)
        
        # Combine: gene_emb (broadcasted) + REE embedding
        # [batch, num_genes, hidden_dim] + [num_genes, hidden_dim] -> [batch, num_genes, hidden_dim]
        x = ree_emb + gene_emb.unsqueeze(0)
        
        # Pass through transformer layers
        for transformer_layer in self.transformer:
            x = transformer_layer(x)  # [batch, num_genes, hidden_dim]
        
        # Project to output
        predictions = self.output_proj(x).squeeze(-1)  # [batch, num_genes]
        
        return predictions


# ============================================================
# Training utilities
# ============================================================

class ExpressionMLMDataset(torch.utils.data.Dataset):
    """Dataset for masked language modeling on expression data."""
    
    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10):
        """
        Args:
            expr_array: [num_genes, num_samples] numpy array
            mask_ratio: fraction of genes to mask
            mask_token: value to use for masked positions
        """
        # Transpose to [num_samples, num_genes]
        self.X = expr_array.T.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]
        
        # Randomly select genes to mask
        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)
        
        # Create masked version
        x_masked = x.copy()
        x_masked[mask_indices] = self.mask_token
        
        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask_indices, dtype=torch.long),
        )


def compute_mlm_loss(predictions, targets, mask_indices):
    """
    Compute MLM loss: MSE on masked positions only.
    
    Args:
        predictions: [batch, num_genes]
        targets: [batch, num_genes]
        mask_indices: [batch_size, max_mask_count] tensor of mask positions
    Returns:
        scalar loss
    """
    loss = 0.0
    count = 0
    
    for i in range(len(predictions)):
        idxs = mask_indices[i]
        if len(idxs) > 0:
            loss += F.mse_loss(predictions[i, idxs], targets[i, idxs])
            count += 1
    
    return loss / max(1, count)


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    num_genes = 16109
    hidden_dim = 256
    
    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=4,
    )
    
    # Test forward pass
    expr_input = torch.randn(batch_size, num_genes)
    output = model(expr_input)
    
    print(f"Input shape: {expr_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

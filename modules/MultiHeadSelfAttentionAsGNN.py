'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Transformer Component as Message Passing
******************************************************************************
'''

import copy
from typing import Union
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from modules import LinearAsGNN, MLPAsGNN

# Multi-Head Self-Attention as GNN
class MultiHeadSelfAttentionAsGNN(MessagePassing):
    """
    Multi-Head Self-Attention as MessagePassing.
    Supports both sequence-first (batch_first=False) and batch-first (batch_first=True) modes.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 bias: bool = True, 
                 dropout: float = 0.0, 
                 batch_first: bool = False
                 ):
        
        # Sum Aggregation
        super(MultiHeadSelfAttentionAsGNN, self).__init__(aggr='add')
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Embed dimension
        self.embed_dim = embed_dim
        
        # Number of heads
        self.num_heads = num_heads
        
        # Head dimension
        self.head_dim = embed_dim // num_heads
        
        # Scaling factor for attention scores
        self.scale = (self.head_dim ** -0.5)
        
        # Batch-first mode
        self.batch_first = batch_first
        
        # Q, K, V Linear Projections
        self.query_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        self.key_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        self.value_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, embed_dim)
            attn_mask: optional boolean mask of shape [B, N, N], True = Masking(NOT Attend)
        Returns:
            torch.Tensor: Output tensor of shape (N, embed_dim)
        """
        device = x.device
        
        # Check input dimensions
        orig_dim = x.dim()
        
        # Sequence-first mode configuration: (seq, batch, E) → (batch, seq, E)
        if not self.batch_first and orig_dim == 3:
            x = x.permute(1, 0, 2)
        if orig_dim == 2:
            # (N, E) → (1, N, E)
            x = x.unsqueeze(0)
        
        # Input shape: (B, N, E)
        B, N, E = x.shape
        
        # Flatten input to (B*N, E)
        x_flat = x.reshape(B * N, E) 

        # Q, K, V
        query = self.query_proj(x_flat) 
        key = self.key_proj(x_flat)
        value = self.value_proj(x_flat)
        
        # Reshape to (B, N, num_heads, head_dim)
        query = query.view(B, N, self.num_heads, self.head_dim)
        key = key.view(B, N, self.num_heads, self.head_dim)
        value = value.view(B, N, self.num_heads, self.head_dim)
        
        # Make Fully Connected Graph
        node_idx = torch.arange(N, dtype=torch.long, device=x.device)
        source_nodes = node_idx.repeat(N)
        target_nodes = node_idx.repeat_interleave(N)
        base_adj_matrix = torch.stack([source_nodes, target_nodes], dim=0)
        
        # Batch Offsets
        edge_idx = torch.cat([base_adj_matrix + b * N for b in range(B)], dim=1)
        
        # Masked Self Attention (Masking Adgacency Matrix)
        # Make Masking
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
            pad_src = key_padding_mask.unsqueeze(1).expand(B, N, N)  # (B, N, N)
            if attn_mask is None:
                attn_mask = pad_src
            else:
                attn_mask = attn_mask.to(device).logical_or(pad_src)
        elif attn_mask is not None:
            attn_mask = attn_mask.to(device)
        
        # Flatten Mask
        mask_flat = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(B, N, N)
            elif attn_mask.dim() == 3:
                pass
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.dim()}D")
      
            # Flatten mask to 1D
            mask_flat = attn_mask.reshape(-1).to(device)
            
        # Compute per-head attention via message passing
        heads_out = []
        for h in range(self.num_heads):
            
            # Extract head-specific query, key, value
            query_h = query[:, :, h, :].reshape(B * N, self.head_dim)
            key_h = key[:, :, h, :].reshape(B * N, self.head_dim)
            value_h = value[:, :, h, :].reshape(B * N, self.head_dim)
            
            # Message Passing per head
            head_out = self.propagate(
                edge_idx,
                query=query_h,
                key=key_h,
                value=value_h,
                size=(B * N, B * N),
                attn_mask=mask_flat
            )
            
            # Store head output
            heads_out.append(head_out)
        
        # Concatenate heads and apply output projection
        out = torch.stack(heads_out, dim=1).view(B * N, E)
        out = self.out_proj(out)
        out = out.view(B, N, E)
        
        # Reshape about batch-first mode
        if not self.batch_first:
            # (seq, batch, embed)
            out = out.permute(1, 0, 2)  
        else:
            # (batch, seq, embed)
            out = out 

        # Reshape back to original dimensions(2D)
        if orig_dim == 2:
            out = out.squeeze(0)
            
        return out
    
    # Message function for attention computation
    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, value_j: torch.Tensor, index: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        query_i:    [#edges, head_d]     (target node   q = Target node)
        key_j:      [#edges, head_d]     (source node   k = Neighbor node)
        value_j:    [#edges, head_d]     (source node   v = Neighbor node)
        index:      [#edges]             (Attention edge weights)
        """
        
        # Compute Attention Edge Scores
        attn_score = (query_i * key_j).sum(dim=-1) * self.scale  
        
        # Apply Masking
        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, float('-inf'))
            
        # Normalize Edge Weights using softmax
        edge_weights = softmax(attn_score, index)
        
        # Apply Dropout if specified
        if self.dropout is not None:
            edge_weights = self.dropout(edge_weights)
        
        # Weighted Message Aggregation
        out = value_j * edge_weights.unsqueeze(-1)
        return out
    
    # Update function(Identity)
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out



# Multi-Head Cross-Attention as GNN
class MultiHeadCrossAttentionAsGNN(MessagePassing):
    """
    Multi-Head CrossAttention as MessagePassing
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 bias: bool = True, 
                 dropout: float = 0.0
                 ):
        
        # Sum Aggregation
        super(MultiHeadCrossAttentionAsGNN, self).__init__(aggr='add')
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Embed dimension
        self.embed_dim = embed_dim
        
        # Number of heads
        self.num_heads = num_heads
        
        # Head dimension
        self.head_dim = embed_dim // num_heads
        
        # Scaling factor for attention scores
        self.scale = (self.head_dim ** -0.5)

        # Q, K, V projections
        self.query_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        self.key_proj   = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        self.value_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = LinearAsGNN(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # tgt: (B, T, E), memory: (B, S, E)
        B, T, E = tgt.shape
        _, S, _ = memory.shape

        # Flatten target and memory
        tgt_flat    = tgt.reshape(B * T, E)
        memory_flat = memory.reshape(B * S, E)

        # Project Q, K, V
        q = self.query_proj(tgt_flat).view(B, T, self.num_heads, self.head_dim)
        k = self.key_proj(memory_flat).view(B, S, self.num_heads, self.head_dim)
        v = self.value_proj(memory_flat).view(B, S, self.num_heads, self.head_dim)

        # Make bipartite edge_idx
        tgt_idx = torch.arange(T, device=tgt.device)
        mem_idx = torch.arange(S, device=tgt.device)
        
        # Source and target nodes for edges
        source_nodes = mem_idx.repeat(T)      
        target_nodes = tgt_idx.repeat_interleave(S) 
        
        # Make edges for each batch
        edges = []
        for b in range(B):
            src_off = b * S
            tgt_off = b * T
            e = torch.stack([source_nodes + src_off, target_nodes + tgt_off], dim=0)
            edges.append(e)
        edge_idx = torch.cat(edges, dim=1)

        # Build attention mask over memory keys if provided
        mask_flat = None
        if memory_key_padding_mask is not None or memory_mask is not None:
            mem_pad = None
            if memory_key_padding_mask is not None:
                mem_pad = memory_key_padding_mask.unsqueeze(1).expand(B, T, S) 
            if memory_mask is not None:
                mm = memory_mask
                if mm.dim() == 2:
                    mm = mm.unsqueeze(0).expand(B, T, S)
                mem_pad = mem_pad.logical_or(mm) if mem_pad is not None else mm
            mask_flat = mem_pad.reshape(-1) if mem_pad is not None else None

        # Compute per-head cross-attention via message passing
        heads_out = []
        for h in range(self.num_heads):
            # Extract head-specific query, key, value
            q_h = q[:, :, h, :].reshape(B * T, self.head_dim)
            k_h = k[:, :, h, :].reshape(B * S, self.head_dim)
            v_h = v[:, :, h, :].reshape(B * S, self.head_dim)
            
            # Message Passing per head
            head_out = self.propagate(
                edge_idx,
                query=q_h,
                key=k_h,
                value=v_h,
                size=(B * S, B * T),
                attn_mask=mask_flat
            )
            
            # Store head output
            heads_out.append(head_out)

        # Concat heads and out projection
        out = torch.stack(heads_out, dim=1).view(B * T, E)
        out = self.out_proj(out).view(B, T, E)
        return out

    # Message function for cross-attention computation
    def message(self, query_i, key_j, value_j, index, attn_mask=None):
        # Attention edge scores
        attn_score = (query_i * key_j).sum(dim=-1) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, float('-inf'))
        
        # Normalize edge weights using softmax
        weights = softmax(attn_score, index)
        
        # Apply dropout if specified
        if self.dropout is not None:
            weights = self.dropout(weights)
        
        # Weighted message aggregation
        return value_j * weights.unsqueeze(-1)


class TransformerEncoderLayerAsGNN(nn.Module):
    """
    Transformer Encoder Layer using MultiHeadSelfAttentionAsGNN,
    with residual connections, LayerNorm, and Feed-Forward.
    Accepts optional attention mask.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 feedforward_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 bias: bool = True,
                 batch_first: bool = False,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        super(TransformerEncoderLayerAsGNN, self).__init__()
        self.batch_first = batch_first
        self.device = torch.device(device)
        
        # Multi-Head Self-Attention
        self.self_attn = MultiHeadSelfAttentionAsGNN(embed_dim, num_heads, bias=bias, dropout=dropout, batch_first=batch_first).to(self.device)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim).to(self.device)
        self.norm2 = nn.LayerNorm(embed_dim).to(self.device)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout).to(self.device)
        self.dropout2 = nn.Dropout(dropout).to(self.device)
        
        # Feed-forward network
        self.ffn = MLPAsGNN(embed_dim, feedforward_dim, embed_dim, activation=activation, bias=bias).to(self.device)
        
        
    def forward(self, 
                x: torch.Tensor, 
                src_mask: torch.Tensor = None, 
                src_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        
        # Device handling
        x = x.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)

        # Self-attention layer
        attn_out = self.self_attn(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # Residual connection and normalization
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward sublayer
        feedforward_out = self.ffn(x)
        
        # Residual connection and normalization
        x = x + self.dropout2(feedforward_out)
        x = self.norm2(x)
        return x


class TransformerEncoderAsGNN(nn.Module):
    """
    Wrapper that stacks multiple TransformerEncoderLayerAsGNN layers,
    mirroring nn.TransformerEncoder.
    """
    def __init__(self,
                 encoder_layer: TransformerEncoderLayerAsGNN,
                 num_layers: int,
                 norm: nn.LayerNorm = None,
                 ):
        super(TransformerEncoderAsGNN, self).__init__()
        
        # Encoder Layer Copy
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        
        # LayerNorm
        if norm is not None:
            self.norm = norm.to(encoder_layer.device)
        else:
            self.norm = None

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None, 
                src_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        
        # Concatenate encoder layers
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class TransformerDecoderLayerAsGNN(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 feedforward_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 bias: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        
        # Multi-Head Self-Attention
        self.self_attn = MultiHeadSelfAttentionAsGNN(embed_dim, 
                                                     num_heads, 
                                                     bias=bias, 
                                                     dropout=dropout, 
                                                     batch_first=True
                                                     ).to(self.device)
        
        # Multi-Head Cross-Attention
        self.cross_attn = MultiHeadCrossAttentionAsGNN(embed_dim, 
                                                       num_heads, 
                                                       bias=bias, 
                                                       dropout=dropout
                                                       ).to(self.device)
        
        # Feed-Forward Network
        self.ffn = MLPAsGNN(embed_dim, 
                            feedforward_dim, 
                            embed_dim, 
                            activation=activation, 
                            bias=bias
                            ).to(self.device)

        # LayerNorm and Dropout
        self.norm1 = nn.LayerNorm(embed_dim).to(self.device)
        self.norm2 = nn.LayerNorm(embed_dim).to(self.device)
        self.norm3 = nn.LayerNorm(embed_dim).to(self.device)
        self.dropout1 = nn.Dropout(dropout).to(self.device)
        self.dropout2 = nn.Dropout(dropout).to(self.device)
        self.dropout3 = nn.Dropout(dropout).to(self.device)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        tgt = tgt.to(self.device)
        memory = memory.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(self.device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.device)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(self.device)

        # Self-attention
        attn_out = self.self_attn(tgt, 
                                  attn_mask=tgt_mask, 
                                  key_padding_mask=tgt_key_padding_mask
                                  )
        
        # Residual connection and normalization
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
        # Cross-attention
        cross_out = self.cross_attn(tgt,
                                    memory,
                                    tgt_mask=None,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=None,
                                    memory_key_padding_mask=memory_key_padding_mask
                                    )
        
        # Residual connection and normalization
        tgt = self.norm2(tgt + self.dropout2(cross_out))
        
        # Feed-forward
        ffn_out = self.ffn(tgt)
        
        # Residual connection and normalization
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt

# Transformer Decoder as GNN
class TransformerDecoderAsGNN(nn.Module):
    def __init__(self,
                 decoder_layer: TransformerDecoderLayerAsGNN,
                 num_layers: int,
                 norm: nn.LayerNorm = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = tgt
        
        # Iterate through decoder layers
        for layer in self.layers:
            x = layer(x, 
                      memory,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask
                      )
        return self.norm(x) if self.norm is not None else x


class TransformerAsGNN(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        
        # Encoder Layer
        memory = self.encoder(src,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask
                              )
        
        # Decoder Layer
        output = self.decoder(tgt,
                              memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    print(f"\n=============== Experiment 1 ===============")
    # ================================================================================
    # 1) Self-Attention as GNN vs nn.MultiheadAttention
    # ================================================================================
    print('\n== 1) Self-Attention 비교 ==')
    B1, N1, E1, H1 = 3, 10, 32, 4
    x1 = torch.randn(B1, N1, E1).to(device)

    lengths = torch.tensor([10, 7, 5], device=device)
    key_padding_mask1 = (torch.arange(N1, device=device)
                         .unsqueeze(0).repeat(B1, 1) >= lengths.unsqueeze(1))

    gnn_attn = MultiHeadSelfAttentionAsGNN(E1, H1, dropout=0.0, bias=True, batch_first=True).to(device)
    std_attn = nn.MultiheadAttention(E1, H1, dropout=0.0, bias=True, batch_first=True).to(device)

    # Copy GNN parameters to standard MultiheadAttention
    std_attn.in_proj_weight.data.copy_(torch.cat([
        gnn_attn.query_proj.weight,
        gnn_attn.key_proj.weight,
        gnn_attn.value_proj.weight
    ], dim=0))
    
    std_attn.in_proj_bias.data.copy_(torch.cat([
        gnn_attn.query_proj.bias,
        gnn_attn.key_proj.bias,
        gnn_attn.value_proj.bias
    ], dim=0))
    
    std_attn.out_proj.weight.data.copy_(gnn_attn.out_proj.weight)
    
    std_attn.out_proj.bias.data.copy_(gnn_attn.out_proj.bias)

    # Compare outputs without mask
    y_gnn1 = gnn_attn(x1)
    y_std1, _ = std_attn(x1, x1, x1, need_weights=False)
    diff0 = (y_gnn1 - y_std1).abs().mean().item()
    print(f'mean diff no mask: {diff0:.3e}')

    # Causal mask for self-attention
    causal_mask1 = torch.triu(torch.ones(N1, N1, dtype=torch.bool), 1).to(device)

    # Compare outputs with causal mask
    y_gnn1_m = gnn_attn(x1, attn_mask=causal_mask1, key_padding_mask=key_padding_mask1)
    y_std1_m, _ = std_attn(x1, x1, x1,
                          attn_mask=causal_mask1,
                          key_padding_mask=key_padding_mask1,
                          need_weights=False)
    diff1 = (y_gnn1_m - y_std1_m).abs().mean().item()
    print(f'mean diff with mask: {diff1:.3e}')

    # ================================================================================
    # 2) TransformerEncoderLayerAsGNN vs nn.TransformerEncoderLayer
    # ================================================================================
    print('\n== 2) TransformerEncoderLayerAsGNN vs nn.TransformerEncoderLayer ==')

    # Configuration
    seq_len = 10
    embed_dim = 32
    heads = 4
    ff_dim = 64
    B2 = 4

    # Random input tensor
    x2 = torch.randn(seq_len, B2, embed_dim).to(device)

    # Transformer Encoder Layer
    gnn_layer = TransformerEncoderLayerAsGNN(
        embed_dim=embed_dim,
        num_heads=heads,
        feedforward_dim=ff_dim,
        dropout=0.0,
        activation='gelu',
        bias=True,
        batch_first=False,
        device=device 
    ).to(device)

    # nn.TransformerEncoderLayer
    standard_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=heads,
        dim_feedforward=ff_dim,
        dropout=0.0,
        activation='gelu',
        batch_first=False,
        norm_first=False
    ).to(device)

    # Copy parameters from GNN layer to standard layer
    q_w, k_w, v_w = (
        gnn_layer.self_attn.query_proj.weight,
        gnn_layer.self_attn.key_proj.weight,
        gnn_layer.self_attn.value_proj.weight
    )
    
    q_b, k_b, v_b = (
        gnn_layer.self_attn.query_proj.bias,
        gnn_layer.self_attn.key_proj.bias,
        gnn_layer.self_attn.value_proj.bias
    )
    
    standard_layer.self_attn.in_proj_weight.data.copy_(
        torch.cat([q_w, k_w, v_w], dim=0)
    )
    
    standard_layer.self_attn.in_proj_bias.data.copy_(
        torch.cat([q_b, k_b, v_b], dim=0)
    )
    
    standard_layer.self_attn.out_proj.weight.data.copy_(
        gnn_layer.self_attn.out_proj.weight
    )
    
    standard_layer.self_attn.out_proj.bias.data.copy_(
        gnn_layer.self_attn.out_proj.bias
    )

    standard_layer.linear1.weight.data.copy_(gnn_layer.ffn.linear1.weight)
    standard_layer.linear1.bias.data.copy_(gnn_layer.ffn.linear1.bias)
    standard_layer.linear2.weight.data.copy_(gnn_layer.ffn.linear2.weight)
    standard_layer.linear2.bias.data.copy_(gnn_layer.ffn.linear2.bias)

    # Compare outputs without mask
    y_gnn2 = gnn_layer(x2)       # (seq_len, B2, embed_dim)
    y_std2 = standard_layer(x2)  # (seq_len, B2, embed_dim)
    diff2 = (y_gnn2 - y_std2).abs().mean().item()
    print(f'mean diff vs nn.TransformerEncoderLayer: {diff2:.3e}')

    # Make causal mask for self-attention
    causal_mask2 = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device)

    # Compare outputs with causal mask
    y_gnn2_m = gnn_layer(x2, src_mask=causal_mask2)
    y_std2_m = standard_layer(x2, src_mask=causal_mask2)
    diff2_m = (y_gnn2_m - y_std2_m).abs().mean().item()
    print(f'mean diff with causal mask: {diff2_m:.3e}')

    # ================================================================================
    # 3) TransformerEncoderAsGNN vs nn.TransformerEncoder
    # ================================================================================
    print('\n== 3) TransformerEncoderAsGNN vs nn.TransformerEncoder ==')

    # Configuration
    embed_dim3 = 32
    num_heads3 = 4
    ff_dim3 = 64
    dropout3 = 0.0
    depth3 = 6
    B3 = 2
    seq_len3 = 10

    # Make Encoder Layer
    base_gnn_layer = TransformerEncoderLayerAsGNN(
        embed_dim=embed_dim3,
        num_heads=num_heads3,
        feedforward_dim=ff_dim3,
        dropout=dropout3,
        activation='gelu',
        batch_first=True,
        device=device
    ).to(device)
    
    # Make LayerNorm for final normalization
    final_norm3 = nn.LayerNorm(embed_dim3).to(device)
    
    # Create TransformerEncoderAsGNN
    encoder_as_gnn = TransformerEncoderAsGNN(
        encoder_layer=base_gnn_layer,
        num_layers=depth3,
        norm=final_norm3
    ).to(device)

    # nn.TransformerEncoder Layer
    standard_layer3 = nn.TransformerEncoderLayer(
        d_model=embed_dim3,
        nhead=num_heads3,
        dim_feedforward=ff_dim3,
        dropout=dropout3,
        activation='gelu',
        batch_first=True,
        norm_first=False
    ).to(device)
    
    # Create nn.TransformerEncoder
    standard_encoder3 = nn.TransformerEncoder(
        encoder_layer=standard_layer3,
        num_layers=depth3,
        norm=final_norm3
    )

    # Copy parameters from GNN encoder to standard encoder
    for i in range(depth3):
        std_ly3 = standard_encoder3.layers[i]
        gnn_ly3 = encoder_as_gnn.layers[i]

        q_w3, k_w3, v_w3 = (
            gnn_ly3.self_attn.query_proj.weight,
            gnn_ly3.self_attn.key_proj.weight,
            gnn_ly3.self_attn.value_proj.weight
        )
        q_b3, k_b3, v_b3 = (
            gnn_ly3.self_attn.query_proj.bias,
            gnn_ly3.self_attn.key_proj.bias,
            gnn_ly3.self_attn.value_proj.bias
        )
        std_ly3.self_attn.in_proj_weight.data.copy_(torch.cat([q_w3, k_w3, v_w3], dim=0))
        std_ly3.self_attn.in_proj_bias.data.copy_(torch.cat([q_b3, k_b3, v_b3], dim=0))
        std_ly3.self_attn.out_proj.weight.data.copy_(gnn_ly3.self_attn.out_proj.weight)
        std_ly3.self_attn.out_proj.bias.data.copy_(gnn_ly3.self_attn.out_proj.bias)

        std_ly3.linear1.weight.data.copy_(gnn_ly3.ffn.linear1.weight)
        std_ly3.linear1.bias.data.copy_(gnn_ly3.ffn.linear1.bias)
        std_ly3.linear2.weight.data.copy_(gnn_ly3.ffn.linear2.weight)
        std_ly3.linear2.bias.data.copy_(gnn_ly3.ffn.linear2.bias)

    # Random input tensor
    x3 = torch.randn(B3, seq_len3, embed_dim3).to(device)
    lengths3 = torch.tensor([10, 7], device=device)
    key_padding_mask3 = torch.arange(seq_len3, device=device).unsqueeze(0).repeat(B3, 1) >= lengths3.unsqueeze(1)
    src_mask3 = torch.triu(torch.ones(seq_len3, seq_len3, dtype=torch.bool), diagonal=1).to(device)

    # Compare outputs without mask
    out_std3 = standard_encoder3(x3, mask=src_mask3, src_key_padding_mask=key_padding_mask3)
    out_gnn3 = encoder_as_gnn(x3, mask=src_mask3, src_key_padding_mask=key_padding_mask3)
    max_diff3 = (out_gnn3 - out_std3).abs().mean().item()
    print(f"mean absolute difference: {max_diff3:.3e}")
    
    # ================================================================================
    # 4) TransformerDecoderLayerAsGNN vs nn.TransformerDecoderLayer
    # ================================================================================
    
    # Configuration
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    dropout = 0.0
    seq_len_tgt = 12
    seq_len_mem = 15
    batch_size = 3

    # Input and mask generation
    torch.manual_seed(42)
    tgt = torch.randn(batch_size, seq_len_tgt, embed_dim).to(device)
    memory = torch.randn(batch_size, seq_len_mem, embed_dim).to(device)

    # causal mask for decoder self-attention (True=block)
    tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
    tgt_key_padding_mask = torch.zeros(batch_size, seq_len_tgt, dtype=torch.bool).to(device)
    mem_key_padding_mask = torch.zeros(batch_size, seq_len_mem, dtype=torch.bool).to(device)

    # Transformer Decoder Layer as GNN
    gnn_dec_layer = TransformerDecoderLayerAsGNN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=ff_dim,
        dropout=dropout,
        activation='relu',
        bias=True,
        device=device
    )

    # Standard Transformer Decoder Layer
    std_dec_layer = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation='relu',
        batch_first=True,
        norm_first=False
    ).to(device)

    # Parameter copying from GNN to standard layer
    q_w, k_w, v_w = (
        gnn_dec_layer.self_attn.query_proj.weight,
        gnn_dec_layer.self_attn.key_proj.weight,
        gnn_dec_layer.self_attn.value_proj.weight
    )
    q_b, k_b, v_b = (
        gnn_dec_layer.self_attn.query_proj.bias,
        gnn_dec_layer.self_attn.key_proj.bias,
        gnn_dec_layer.self_attn.value_proj.bias
    )
    std_dec_layer.self_attn.in_proj_weight.data.copy_(
        torch.cat([q_w, k_w, v_w], dim=0)
    )
    std_dec_layer.self_attn.in_proj_bias.data.copy_(
        torch.cat([q_b, k_b, v_b], dim=0)
    )
    std_dec_layer.self_attn.out_proj.weight.data.copy_(gnn_dec_layer.self_attn.out_proj.weight)
    std_dec_layer.self_attn.out_proj.bias.data.copy_(gnn_dec_layer.self_attn.out_proj.bias)

    q_w, k_w, v_w = (
        gnn_dec_layer.cross_attn.query_proj.weight,
        gnn_dec_layer.cross_attn.key_proj.weight,
        gnn_dec_layer.cross_attn.value_proj.weight
    )
    q_b, k_b, v_b = (
        gnn_dec_layer.cross_attn.query_proj.bias,
        gnn_dec_layer.cross_attn.key_proj.bias,
        gnn_dec_layer.cross_attn.value_proj.bias
    )
    std_dec_layer.multihead_attn.in_proj_weight.data.copy_(
        torch.cat([q_w, k_w, v_w], dim=0)
    )
    std_dec_layer.multihead_attn.in_proj_bias.data.copy_(
        torch.cat([q_b, k_b, v_b], dim=0)
    )
    std_dec_layer.multihead_attn.out_proj.weight.data.copy_(gnn_dec_layer.cross_attn.out_proj.weight)
    std_dec_layer.multihead_attn.out_proj.bias.data.copy_(gnn_dec_layer.cross_attn.out_proj.bias)

    std_dec_layer.linear1.weight.data.copy_(gnn_dec_layer.ffn.linear1.weight)
    std_dec_layer.linear1.bias.data.copy_(gnn_dec_layer.ffn.linear1.bias)
    std_dec_layer.linear2.weight.data.copy_(gnn_dec_layer.ffn.linear2.weight)
    std_dec_layer.linear2.bias.data.copy_(gnn_dec_layer.ffn.linear2.bias)

    # Calculate outputs
    y_gnn = gnn_dec_layer(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=None,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=mem_key_padding_mask
    )
    y_std = std_dec_layer(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=mem_key_padding_mask
    )

    diff = (y_gnn - y_std).abs().mean().item()
    print(f"mean abs diff between GNN‐DecoderLayer and nn.TransformerDecoderLayer: {diff:.3e}")




    # ================================================================================
    # 5) TransformerDecoderLayerAsGNN vs nn.TransformerDecoderLayer without mask
    # ================================================================================
    print('\n== TransformerDecoderLayerAsGNN vs nn.TransformerDecoderLayer ==')

    # Configuration
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    dropout = 0.0
    seq_len_tgt = 12
    seq_len_mem = 15
    batch_size = 3

    # Make input tensors
    torch.manual_seed(42)
    tgt = torch.randn(batch_size, seq_len_tgt, embed_dim).to(device)      # (B, T, E)
    memory = torch.randn(batch_size, seq_len_mem, embed_dim).to(device)   # (B, S, E)

    # causal mask for decoder self-attention
    tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
    
    # padding masks
    tgt_key_padding_mask = torch.zeros(batch_size, seq_len_tgt, dtype=torch.bool).to(device)
    mem_key_padding_mask = torch.zeros(batch_size, seq_len_mem, dtype=torch.bool).to(device)

    # Make GNN-based Transformer Decoder Layer
    gnn_dec_layer = TransformerDecoderLayerAsGNN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=ff_dim,
        dropout=dropout,
        activation='relu',
        bias=True,
        device=device
    )

    # Standard Transformer Decoder Layer
    std_dec_layer = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation='relu',
        batch_first=True,
        norm_first=False
    ).to(device) 

    # Copy parameters from GNN layer to standard layer
    q_w, k_w, v_w = (
        gnn_dec_layer.self_attn.query_proj.weight,
        gnn_dec_layer.self_attn.key_proj.weight,
        gnn_dec_layer.self_attn.value_proj.weight
    )
    q_b, k_b, v_b = (
        gnn_dec_layer.self_attn.query_proj.bias,
        gnn_dec_layer.self_attn.key_proj.bias,
        gnn_dec_layer.self_attn.value_proj.bias
    )
    std_dec_layer.self_attn.in_proj_weight.data.copy_(
        torch.cat([q_w, k_w, v_w], dim=0)
    )
    std_dec_layer.self_attn.in_proj_bias.data.copy_(
        torch.cat([q_b, k_b, v_b], dim=0)
    )
    std_dec_layer.self_attn.out_proj.weight.data.copy_(gnn_dec_layer.self_attn.out_proj.weight)
    std_dec_layer.self_attn.out_proj.bias.data.copy_(gnn_dec_layer.self_attn.out_proj.bias)

    q_w, k_w, v_w = (
        gnn_dec_layer.cross_attn.query_proj.weight,
        gnn_dec_layer.cross_attn.key_proj.weight,
        gnn_dec_layer.cross_attn.value_proj.weight
    )
    q_b, k_b, v_b = (
        gnn_dec_layer.cross_attn.query_proj.bias,
        gnn_dec_layer.cross_attn.key_proj.bias,
        gnn_dec_layer.cross_attn.value_proj.bias
    )
    std_dec_layer.multihead_attn.in_proj_weight.data.copy_(
        torch.cat([q_w, k_w, v_w], dim=0)
    )
    std_dec_layer.multihead_attn.in_proj_bias.data.copy_(
        torch.cat([q_b, k_b, v_b], dim=0)
    )
    std_dec_layer.multihead_attn.out_proj.weight.data.copy_(gnn_dec_layer.cross_attn.out_proj.weight)
    std_dec_layer.multihead_attn.out_proj.bias.data.copy_(gnn_dec_layer.cross_attn.out_proj.bias)

    std_dec_layer.linear1.weight.data.copy_(gnn_dec_layer.ffn.linear1.weight)
    std_dec_layer.linear1.bias.data.copy_(gnn_dec_layer.ffn.linear1.bias)
    std_dec_layer.linear2.weight.data.copy_(gnn_dec_layer.ffn.linear2.weight)
    std_dec_layer.linear2.bias.data.copy_(gnn_dec_layer.ffn.linear2.bias)

    # Calculate outputs
    y_gnn2 = gnn_dec_layer(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=None,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=mem_key_padding_mask
    )

    y_std2 = std_dec_layer(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=mem_key_padding_mask
    )

    diff2 = (y_gnn2 - y_std2).abs().mean().item()
    print(f"mean abs diff between GNN‐DecoderLayer and nn.TransformerDecoderLayer: {diff2:.3e}")

    # ================================================================================
    # 6) TransformerDecoderAsGNN vs nn.TransformerDecoder with mask
    # ================================================================================
    print('\n== TransformerDecoderAsGNN vs nn.TransformerDecoder ==')

    # Configuration
    embed_dim3 = 32
    num_heads3 = 4
    ff_dim3 = 64
    dropout3 = 0.0
    depth3 = 6
    batch_first3 = True

    # Make GNN-based Transformer Decoder Layer
    base_gnn_dec_layer = TransformerDecoderLayerAsGNN(
        embed_dim=embed_dim3,
        num_heads=num_heads3,
        feedforward_dim=ff_dim3,
        dropout=dropout3,
        activation='gelu',
        bias=True,
        device=device
    )
    
    # Make LayerNorm for final normalization
    final_norm3 = nn.LayerNorm(embed_dim3).to(device)
    
    # Create TransformerDecoderAsGNN
    decoder_as_gnn = TransformerDecoderAsGNN(
        decoder_layer=base_gnn_dec_layer,
        num_layers=depth3,
        norm=final_norm3
    )

    # Standard Transformer Decoder Layer
    std_dec_ly3 = nn.TransformerDecoderLayer(
        d_model=embed_dim3,
        nhead=num_heads3,
        dim_feedforward=ff_dim3,
        dropout=dropout3,
        activation='gelu',
        batch_first=batch_first3,
        norm_first=False
    ).to(device)
    
    # Create nn.TransformerDecoder
    standard_decoder3 = nn.TransformerDecoder(
        decoder_layer=std_dec_ly3,
        num_layers=depth3,
        norm=final_norm3
    )

    # Copy parameters from GNN decoder to standard decoder
    for i in range(depth3):
        std_ly3 = standard_decoder3.layers[i]
        gnn_ly3 = decoder_as_gnn.layers[i]

        q_w3, k_w3, v_w3 = (
            gnn_ly3.self_attn.query_proj.weight,
            gnn_ly3.self_attn.key_proj.weight,
            gnn_ly3.self_attn.value_proj.weight
        )
        q_b3, k_b3, v_b3 = (
            gnn_ly3.self_attn.query_proj.bias,
            gnn_ly3.self_attn.key_proj.bias,
            gnn_ly3.self_attn.value_proj.bias
        )
        std_ly3.self_attn.in_proj_weight.data.copy_(torch.cat([q_w3, k_w3, v_w3], dim=0))
        std_ly3.self_attn.in_proj_bias.data.copy_(torch.cat([q_b3, k_b3, v_b3], dim=0))
        std_ly3.self_attn.out_proj.weight.data.copy_(gnn_ly3.self_attn.out_proj.weight)
        std_ly3.self_attn.out_proj.bias.data.copy_(gnn_ly3.self_attn.out_proj.bias)

        q_w3c, k_w3c, v_w3c = (
            gnn_ly3.cross_attn.query_proj.weight,
            gnn_ly3.cross_attn.key_proj.weight,
            gnn_ly3.cross_attn.value_proj.weight
        )
        q_b3c, k_b3c, v_b3c = (
            gnn_ly3.cross_attn.query_proj.bias,
            gnn_ly3.cross_attn.key_proj.bias,
            gnn_ly3.cross_attn.value_proj.bias
        )
        std_ly3.multihead_attn.in_proj_weight.data.copy_(torch.cat([q_w3c, k_w3c, v_w3c], dim=0))
        std_ly3.multihead_attn.in_proj_bias.data.copy_(torch.cat([q_b3c, k_b3c, v_b3c], dim=0))
        std_ly3.multihead_attn.out_proj.weight.data.copy_(gnn_ly3.cross_attn.out_proj.weight)
        std_ly3.multihead_attn.out_proj.bias.data.copy_(gnn_ly3.cross_attn.out_proj.bias)

        std_ly3.linear1.weight.data.copy_(gnn_ly3.ffn.linear1.weight)
        std_ly3.linear1.bias.data.copy_(gnn_ly3.ffn.linear1.bias)
        std_ly3.linear2.weight.data.copy_(gnn_ly3.ffn.linear2.weight)
        std_ly3.linear2.bias.data.copy_(gnn_ly3.ffn.linear2.bias)

    # Make Mask and Padding Mask
    torch.manual_seed(0)
    B4, S_tgt4, S_mem4 = 2, 10, 12
    tgt4 = torch.randn(B4, S_tgt4, embed_dim3).to(device)
    memory4 = torch.randn(B4, S_mem4, embed_dim3).to(device)

    tgt_mask4 = torch.triu(torch.ones(S_tgt4, S_tgt4, dtype=torch.bool), diagonal=1).to(device)
    tgt_key_padding_mask4 = torch.zeros(B4, S_tgt4, dtype=torch.bool).to(device)
    mem_key_padding_mask4 = torch.zeros(B4, S_mem4, dtype=torch.bool).to(device)

    # Calculate outputs
    out_gnn4 = decoder_as_gnn(
        tgt4,
        memory4,
        tgt_mask=tgt_mask4,
        memory_mask=None,
        tgt_key_padding_mask=tgt_key_padding_mask4,
        memory_key_padding_mask=mem_key_padding_mask4
    )
    out_std4 = standard_decoder3(
        tgt4,
        memory4,
        tgt_mask=tgt_mask4,
        memory_mask=None,
        tgt_key_padding_mask=tgt_key_padding_mask4,
        memory_key_padding_mask=mem_key_padding_mask4
    )
    max_diff4 = (out_gnn4 - out_std4).abs().mean().item()
    print(f"mean absolute difference: {max_diff4:.3e}")
    
    
    # ================================================================================
    # 7) TransformerAsGNN vs nn.Transformer (device 반영)
    # ================================================================================
    print('\n== TransformerAsGNN vs nn.Transformer ==')

    # Configuration
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    dropout = 0.0
    depth = 6
    batch_first = True

    # Make GNN-based Transformer Encoder Layer
    base_enc = TransformerEncoderLayerAsGNN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=ff_dim,
        dropout=dropout,
        activation='gelu',
        batch_first=batch_first,
        device=device
    )
    
    # LayerNorm for final normalization
    final_norm_enc = nn.LayerNorm(embed_dim).to(device)
    
    # Make GNN-based Transformer Encoder
    enc_as_gnn = TransformerEncoderAsGNN(
        encoder_layer=base_enc,
        num_layers=depth,
        norm=final_norm_enc
    )

    # Make GNN-based Transformer Decoder Layer
    base_dec = TransformerDecoderLayerAsGNN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        feedforward_dim=ff_dim,
        dropout=dropout,
        activation='gelu',
        bias=True,
        device=device
    )
    
    # LayerNorm for final normalization
    final_norm_dec = nn.LayerNorm(embed_dim).to(device)
    
    # Make GNN-based Transformer Decoder
    dec_as_gnn = TransformerDecoderAsGNN(
        decoder_layer=base_dec,
        num_layers=depth,
        norm=final_norm_dec
    )

    # Wrapper
    trans_gnn = TransformerAsGNN(enc_as_gnn, dec_as_gnn)

    # Standard Transformer
    std_trans = nn.Transformer(
        d_model=embed_dim,
        nhead=num_heads,
        num_encoder_layers=depth,
        num_decoder_layers=depth,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation='gelu',
        batch_first=batch_first,
        norm_first=False
    ).to(device)

    # Copy parameters from GNN to standard Transformer
    for i in range(depth):
        std_ly = std_trans.encoder.layers[i]
        gnn_ly = enc_as_gnn.layers[i]

        # Self‐Attn (qkv + out_proj)
        q_w, k_w, v_w = (
            gnn_ly.self_attn.query_proj.weight,
            gnn_ly.self_attn.key_proj.weight,
            gnn_ly.self_attn.value_proj.weight
        )
        q_b, k_b, v_b = (
            gnn_ly.self_attn.query_proj.bias,
            gnn_ly.self_attn.key_proj.bias,
            gnn_ly.self_attn.value_proj.bias
        )
        std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
        std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
        std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
        std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

        # FFN
        std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
        std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
        std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
        std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

    for i in range(depth):
        std_ly = std_trans.decoder.layers[i]
        gnn_ly = dec_as_gnn.layers[i]

        # Self‐Attn
        q_w, k_w, v_w = (
            gnn_ly.self_attn.query_proj.weight,
            gnn_ly.self_attn.key_proj.weight,
            gnn_ly.self_attn.value_proj.weight
        )
        q_b, k_b, v_b = (
            gnn_ly.self_attn.query_proj.bias,
            gnn_ly.self_attn.key_proj.bias,
            gnn_ly.self_attn.value_proj.bias
        )
        std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
        std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
        std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
        std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

        # Cross‐Attn
        q_wc, k_wc, v_wc = (
            gnn_ly.cross_attn.query_proj.weight,
            gnn_ly.cross_attn.key_proj.weight,
            gnn_ly.cross_attn.value_proj.weight
        )
        q_bc, k_bc, v_bc = (
            gnn_ly.cross_attn.query_proj.bias,
            gnn_ly.cross_attn.key_proj.bias,
            gnn_ly.cross_attn.value_proj.bias
        )
        std_ly.multihead_attn.in_proj_weight.data.copy_(torch.cat([q_wc, k_wc, v_wc], dim=0))
        std_ly.multihead_attn.in_proj_bias.data.copy_(torch.cat([q_bc, k_bc, v_bc], dim=0))
        std_ly.multihead_attn.out_proj.weight.data.copy_(gnn_ly.cross_attn.out_proj.weight)
        std_ly.multihead_attn.out_proj.bias.data.copy_(gnn_ly.cross_attn.out_proj.bias)

        # FFN
        std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
        std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
        std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
        std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

    # Random input tensors
    torch.manual_seed(0)
    B, S_src, S_tgt = 2, 10, 8
    src = torch.randn(B, S_src, embed_dim).to(device)
    tgt = torch.randn(B, S_tgt, embed_dim).to(device)

    tgt_mask = torch.triu(torch.ones(S_tgt, S_tgt, dtype=torch.bool), 1).to(device)
    src_kpm = torch.zeros(B, S_src, dtype=torch.bool).to(device)
    tgt_kpm = torch.zeros(B, S_tgt, dtype=torch.bool).to(device)

    # Compare outputs
    out_gnn = trans_gnn(
        src, tgt,
        src_mask=None,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_kpm,
        tgt_key_padding_mask=tgt_kpm,
        memory_mask=None,
        memory_key_padding_mask=src_kpm
    )
    out_std = std_trans(
        src, tgt,
        src_mask=None,
        tgt_mask=tgt_mask,
        memory_mask=None,
        src_key_padding_mask=src_kpm,
        tgt_key_padding_mask=tgt_kpm,
        memory_key_padding_mask=src_kpm
    )
    max_diff = (out_gnn - out_std).abs().mean().item()
    print(f"mean absolute difference: {max_diff:.3e}")


    # ───────────────────────────────────────────────────────────
    # Experiment 2
    # ───────────────────────────────────────────────────────────
    print(f"\n=============== Experiment 2 ===============")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    BATCH_SIZE = 1
    EMBED_DIM  = 64
    NUM_HEADS  = 8
    FF_DIM     = 256    # 보통 feedforward는 4 * embed_dim
    DEPTH      = 2      # Encoder/Decoder 쌓을 레이어 수 (예시로 2개)
    SEQ_LEN_LIST = [10, 20, 50]  # 비교를 위해 시퀀스 길이 바꿔가며 실험
    NUM_ITERS = 10000

    # ───────────────────────────────────────────────────────────
    # 1) MultiHeadSelfAttentionAsGNN vs nn.MultiheadAttention
    # ───────────────────────────────────────────────────────────
    for seq_len in SEQ_LEN_LIST:
        print(f"\n=== [Self-Attn 비교] seq_len={seq_len} ===")
        gnn_mha = MultiHeadSelfAttentionAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.0,
            bias=True,
            batch_first=True
        ).to(device)

        std_mha = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.0,
            bias=True,
            batch_first=True
        ).to(device)

        with torch.no_grad():
            std_mha.in_proj_weight.copy_(torch.cat([
                gnn_mha.query_proj.weight,
                gnn_mha.key_proj.weight,
                gnn_mha.value_proj.weight
            ], dim=0))
            std_mha.in_proj_bias.copy_(torch.cat([
                gnn_mha.query_proj.bias,
                gnn_mha.key_proj.bias,
                gnn_mha.value_proj.bias
            ], dim=0))
            std_mha.out_proj.weight.copy_(gnn_mha.out_proj.weight)
            std_mha.out_proj.bias.copy_(gnn_mha.out_proj.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            x = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
            y_gnn = gnn_mha(x)
            y_std, _ = std_mha(x, x, x, need_weights=False)
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        # 출력 shape 확인
        x_sample = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
        out_shape_gnn = tuple(gnn_mha(x_sample).shape)
        out_shape_std = tuple(std_mha(x_sample, x_sample, x_sample, need_weights=False)[0].shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    # ───────────────────────────────────────────────────────────
    # 2) TransformerEncoderLayerAsGNN vs nn.TransformerEncoderLayer
    # ───────────────────────────────────────────────────────────
    for seq_len in SEQ_LEN_LIST:
        print(f"\n=== [EncoderLayer 비교] seq_len={seq_len} ===")
        gnn_enc = TransformerEncoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='relu',
            bias=True,
            batch_first=True,
            device=device
        ).to(device)

        std_enc = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation='relu',
            batch_first=True,
            norm_first=False
        ).to(device)

        with torch.no_grad():
            # Self-Attn
            q_w, k_w, v_w = (
                gnn_enc.self_attn.query_proj.weight,
                gnn_enc.self_attn.key_proj.weight,
                gnn_enc.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_enc.self_attn.query_proj.bias,
                gnn_enc.self_attn.key_proj.bias,
                gnn_enc.self_attn.value_proj.bias
            )
            std_enc.self_attn.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_enc.self_attn.in_proj_bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_enc.self_attn.out_proj.weight.copy_(gnn_enc.self_attn.out_proj.weight)
            std_enc.self_attn.out_proj.bias.copy_(gnn_enc.self_attn.out_proj.bias)

            # FFN
            std_enc.linear1.weight.copy_(gnn_enc.ffn.linear1.weight)
            std_enc.linear1.bias.copy_(gnn_enc.ffn.linear1.bias)
            std_enc.linear2.weight.copy_(gnn_enc.ffn.linear2.weight)
            std_enc.linear2.bias.copy_(gnn_enc.ffn.linear2.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            x = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
            y_gnn = gnn_enc(x)
            y_std = std_enc(x)
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        x_sample = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
        out_shape_gnn = tuple(gnn_enc(x_sample).shape)
        out_shape_std = tuple(std_enc(x_sample).shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    # ───────────────────────────────────────────────────────────
    # 3) TransformerDecoderLayerAsGNN vs nn.TransformerDecoderLayer
    # ───────────────────────────────────────────────────────────
    for seq_len_tgt in SEQ_LEN_LIST:
        seq_len_mem = seq_len_tgt + 5 
        print(f"\n=== [DecoderLayer 비교] tgt_len={seq_len_tgt}, mem_len={seq_len_mem} ===")
        gnn_dec = TransformerDecoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='relu',
            bias=True,
            device=device
        ).to(device)

        std_dec = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation='relu',
            batch_first=True,
            norm_first=False
        ).to(device)

        with torch.no_grad():
            q_w, k_w, v_w = (
                gnn_dec.self_attn.query_proj.weight,
                gnn_dec.self_attn.key_proj.weight,
                gnn_dec.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_dec.self_attn.query_proj.bias,
                gnn_dec.self_attn.key_proj.bias,
                gnn_dec.self_attn.value_proj.bias
            )
            std_dec.self_attn.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_dec.self_attn.in_proj_bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_dec.self_attn.out_proj.weight.copy_(gnn_dec.self_attn.out_proj.weight)
            std_dec.self_attn.out_proj.bias.copy_(gnn_dec.self_attn.out_proj.bias)

            # Cross-Attn 복사
            q_w, k_w, v_w = (
                gnn_dec.cross_attn.query_proj.weight,
                gnn_dec.cross_attn.key_proj.weight,
                gnn_dec.cross_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_dec.cross_attn.query_proj.bias,
                gnn_dec.cross_attn.key_proj.bias,
                gnn_dec.cross_attn.value_proj.bias
            )
            std_dec.multihead_attn.in_proj_weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_dec.multihead_attn.in_proj_bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_dec.multihead_attn.out_proj.weight.copy_(gnn_dec.cross_attn.out_proj.weight)
            std_dec.multihead_attn.out_proj.bias.copy_(gnn_dec.cross_attn.out_proj.bias)

            # FFN 복사
            std_dec.linear1.weight.copy_(gnn_dec.ffn.linear1.weight)
            std_dec.linear1.bias.copy_(gnn_dec.ffn.linear1.bias)
            std_dec.linear2.weight.copy_(gnn_dec.ffn.linear2.weight)
            std_dec.linear2.bias.copy_(gnn_dec.ffn.linear2.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            tgt = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
            memory = torch.randn(BATCH_SIZE, seq_len_mem, EMBED_DIM, device=device)
            # causal mask
            tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
            tgt_kpm = torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device)
            mem_kpm = torch.zeros(BATCH_SIZE, seq_len_mem, dtype=torch.bool).to(device)

            y_gnn = gnn_dec(tgt, memory, tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=tgt_kpm,
                            memory_key_padding_mask=mem_kpm)
            y_std = std_dec(tgt, memory, tgt_mask=tgt_mask,
                            memory_key_padding_mask=mem_kpm)
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        tgt_sample = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
        mem_sample = torch.randn(BATCH_SIZE, seq_len_mem, EMBED_DIM, device=device)
        tgt_mask_s = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
        y_gnn_s = gnn_dec(tgt_sample, mem_sample, tgt_mask=tgt_mask_s,
                          memory_mask=None,
                          tgt_key_padding_mask=torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device),
                          memory_key_padding_mask=torch.zeros(BATCH_SIZE, seq_len_mem, dtype=torch.bool).to(device))
        y_std_s = std_dec(tgt_sample, mem_sample, tgt_mask=tgt_mask_s,
                          memory_key_padding_mask=torch.zeros(BATCH_SIZE, seq_len_mem, dtype=torch.bool).to(device))
        out_shape_gnn = tuple(y_gnn_s.shape)
        out_shape_std = tuple(y_std_s.shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    # ───────────────────────────────────────────────────────────
    # 4) TransformerEncoderAsGNN vs nn.TransformerEncoder
    # ───────────────────────────────────────────────────────────
    for seq_len in SEQ_LEN_LIST:
        print(f"\n=== [Encoder 전체 비교] seq_len={seq_len}, depth={DEPTH} ===")
        base_gnn_enc = TransformerEncoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='gelu',
            bias=True,
            batch_first=True,
            device=device
        ).to(device)
        final_norm_enc = nn.LayerNorm(EMBED_DIM).to(device)
        enc_as_gnn = TransformerEncoderAsGNN(
            encoder_layer=base_gnn_enc,
            num_layers=DEPTH,
            norm=final_norm_enc
        ).to(device)

        std_layer_list = []
        for _ in range(DEPTH):
            l = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=False
            ).to(device)
            std_layer_list.append(l)
        std_enc = nn.TransformerEncoder(
            encoder_layer=std_layer_list[0],
            num_layers=DEPTH,
            norm=final_norm_enc
        ).to(device)

        # Layer Parameter Copy
        for i in range(DEPTH):
            std_ly = std_enc.layers[i]
            gnn_ly = enc_as_gnn.layers[i]
            
            q_w, k_w, v_w = (
                gnn_ly.self_attn.query_proj.weight,
                gnn_ly.self_attn.key_proj.weight,
                gnn_ly.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.self_attn.query_proj.bias,
                gnn_ly.self_attn.key_proj.bias,
                gnn_ly.self_attn.value_proj.bias
            )
            std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
            std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

            # FFN 복사
            std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
            std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
            std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
            std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            x = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
            # causal mask
            src_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device)
            src_kpm = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool).to(device)

            y_gnn = enc_as_gnn(x, mask=src_mask, src_key_padding_mask=src_kpm)
            y_std = std_enc(x, mask=src_mask, src_key_padding_mask=src_kpm)
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        x_samp = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)
        src_mask_s = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(device)
        src_kpm_s = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool).to(device)
        y_gnn_s = enc_as_gnn(x_samp, mask=src_mask_s, src_key_padding_mask=src_kpm_s)
        y_std_s = std_enc(x_samp, mask=src_mask_s, src_key_padding_mask=src_kpm_s)
        out_shape_gnn = tuple(y_gnn_s.shape)
        out_shape_std = tuple(y_std_s.shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    # ───────────────────────────────────────────────────────────
    # 5) TransformerDecoderAsGNN vs nn.TransformerDecoder
    # ───────────────────────────────────────────────────────────
    for seq_len_tgt in SEQ_LEN_LIST:
        seq_len_mem = seq_len_tgt + 5
        print(f"\n=== [Decoder 전체 비교] tgt_len={seq_len_tgt}, mem_len={seq_len_mem}, depth={DEPTH} ===")
        base_gnn_dec = TransformerDecoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='gelu',
            bias=True,
            device=device
        ).to(device)
        final_norm_dec = nn.LayerNorm(EMBED_DIM).to(device)
        dec_as_gnn = TransformerDecoderAsGNN(
            decoder_layer=base_gnn_dec,
            num_layers=DEPTH,
            norm=final_norm_dec
        ).to(device)

        # Transformer Decoder Layer
        std_layer_list = []
        for _ in range(DEPTH):
            l = nn.TransformerDecoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=False
            ).to(device)
            std_layer_list.append(l)
        std_dec = nn.TransformerDecoder(
            decoder_layer=std_layer_list[0],
            num_layers=DEPTH,
            norm=final_norm_dec
        ).to(device)

        # Copy Parameter
        for i in range(DEPTH):
            std_ly = std_dec.layers[i]
            gnn_ly = dec_as_gnn.layers[i]

            q_w, k_w, v_w = (
                gnn_ly.self_attn.query_proj.weight,
                gnn_ly.self_attn.key_proj.weight,
                gnn_ly.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.self_attn.query_proj.bias,
                gnn_ly.self_attn.key_proj.bias,
                gnn_ly.self_attn.value_proj.bias
            )
            std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
            std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

            q_w, k_w, v_w = (
                gnn_ly.cross_attn.query_proj.weight,
                gnn_ly.cross_attn.key_proj.weight,
                gnn_ly.cross_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.cross_attn.query_proj.bias,
                gnn_ly.cross_attn.key_proj.bias,
                gnn_ly.cross_attn.value_proj.bias
            )
            std_ly.multihead_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.multihead_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.multihead_attn.out_proj.weight.data.copy_(gnn_ly.cross_attn.out_proj.weight)
            std_ly.multihead_attn.out_proj.bias.data.copy_(gnn_ly.cross_attn.out_proj.bias)

            std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
            std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
            std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
            std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            tgt = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
            memory = torch.randn(BATCH_SIZE, seq_len_mem, EMBED_DIM, device=device)
            tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
            tgt_kpm = torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device)
            mem_kpm = torch.zeros(BATCH_SIZE, seq_len_mem, dtype=torch.bool).to(device)

            y_gnn = dec_as_gnn(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm
            )
            y_std = std_dec(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_kpm
            )
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        tgt_s = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
        mem_s = torch.randn(BATCH_SIZE, seq_len_mem, EMBED_DIM, device=device)
        tgt_mask_s = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
        tgt_kpm_s = torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device)
        mem_kpm_s = torch.zeros(BATCH_SIZE, seq_len_mem, dtype=torch.bool).to(device)
        y_gnn_s = dec_as_gnn(
            tgt_s, mem_s,
            tgt_mask=tgt_mask_s,
            memory_mask=None,
            tgt_key_padding_mask=tgt_kpm_s,
            memory_key_padding_mask=mem_kpm_s
        )
        y_std_s = std_dec(
            tgt_s, mem_s,
            tgt_mask=tgt_mask_s,
            memory_key_padding_mask=mem_kpm_s
        )
        out_shape_gnn = tuple(y_gnn_s.shape)
        out_shape_std = tuple(y_std_s.shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")

    # ───────────────────────────────────────────────────────────
    # 6) TransformerAsGNN vs nn.Transformer
    # ───────────────────────────────────────────────────────────
    for seq_len_src in SEQ_LEN_LIST:
        seq_len_tgt = seq_len_src // 2  
        print(f"\n=== [Transformer 전체 비교] src_len={seq_len_src}, tgt_len={seq_len_tgt}, depth={DEPTH} ===")
        
        # Encoder
        base_gnn_enc = TransformerEncoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            device=device
        ).to(device)
        final_norm_enc = nn.LayerNorm(EMBED_DIM).to(device)
        enc_as_gnn = TransformerEncoderAsGNN(
            encoder_layer=base_gnn_enc,
            num_layers=DEPTH,
            norm=final_norm_enc
        ).to(device)

        # Decoder
        base_gnn_dec = TransformerDecoderLayerAsGNN(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            feedforward_dim=FF_DIM,
            dropout=0.0,
            activation='gelu',
            bias=True,
            device=device
        ).to(device)
        final_norm_dec = nn.LayerNorm(EMBED_DIM).to(device)
        dec_as_gnn = TransformerDecoderAsGNN(
            decoder_layer=base_gnn_dec,
            num_layers=DEPTH,
            norm=final_norm_dec
        ).to(device)

        trans_gnn = TransformerAsGNN(enc_as_gnn, dec_as_gnn).to(device)

        # Make Transformer Layer
        std_trans = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            num_encoder_layers=DEPTH,
            num_decoder_layers=DEPTH,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        ).to(device)

        # Encoder Parameter Copy
        for i in range(DEPTH):
            std_ly = std_trans.encoder.layers[i]
            gnn_ly = enc_as_gnn.layers[i]

            q_w, k_w, v_w = (
                gnn_ly.self_attn.query_proj.weight,
                gnn_ly.self_attn.key_proj.weight,
                gnn_ly.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.self_attn.query_proj.bias,
                gnn_ly.self_attn.key_proj.bias,
                gnn_ly.self_attn.value_proj.bias
            )
            std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
            std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

            std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
            std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
            std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
            std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

        # Decoder Parameter Copy
        for i in range(DEPTH):
            std_ly = std_trans.decoder.layers[i]
            gnn_ly = dec_as_gnn.layers[i]

            # Self-Attn
            q_w, k_w, v_w = (
                gnn_ly.self_attn.query_proj.weight,
                gnn_ly.self_attn.key_proj.weight,
                gnn_ly.self_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.self_attn.query_proj.bias,
                gnn_ly.self_attn.key_proj.bias,
                gnn_ly.self_attn.value_proj.bias
            )
            std_ly.self_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.self_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.self_attn.out_proj.weight.data.copy_(gnn_ly.self_attn.out_proj.weight)
            std_ly.self_attn.out_proj.bias.data.copy_(gnn_ly.self_attn.out_proj.bias)

            # Cross-Attn
            q_w, k_w, v_w = (
                gnn_ly.cross_attn.query_proj.weight,
                gnn_ly.cross_attn.key_proj.weight,
                gnn_ly.cross_attn.value_proj.weight
            )
            q_b, k_b, v_b = (
                gnn_ly.cross_attn.query_proj.bias,
                gnn_ly.cross_attn.key_proj.bias,
                gnn_ly.cross_attn.value_proj.bias
            )
            std_ly.multihead_attn.in_proj_weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            std_ly.multihead_attn.in_proj_bias.data.copy_(torch.cat([q_b, k_b, v_b], dim=0))
            std_ly.multihead_attn.out_proj.weight.data.copy_(gnn_ly.cross_attn.out_proj.weight)
            std_ly.multihead_attn.out_proj.bias.data.copy_(gnn_ly.cross_attn.out_proj.bias)

            # FFN
            std_ly.linear1.weight.data.copy_(gnn_ly.ffn.linear1.weight)
            std_ly.linear1.bias.data.copy_(gnn_ly.ffn.linear1.bias)
            std_ly.linear2.weight.data.copy_(gnn_ly.ffn.linear2.weight)
            std_ly.linear2.bias.data.copy_(gnn_ly.ffn.linear2.bias)

        total_mae = 0.0
        for _ in range(NUM_ITERS):
            src = torch.randn(BATCH_SIZE, seq_len_src, EMBED_DIM, device=device)
            tgt = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
            tgt_mask = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
            src_kpm = torch.zeros(BATCH_SIZE, seq_len_src, dtype=torch.bool).to(device)
            tgt_kpm = torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device)
            mem_kpm = src_kpm

            y_gnn = trans_gnn(
                src, tgt,
                src_mask=None,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_kpm,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm
            )
            y_std = std_trans(
                src, tgt,
                src_mask=None,
                tgt_mask=tgt_mask,
                memory_mask=None,
                src_key_padding_mask=src_kpm,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm
            )
            mae = (y_gnn - y_std).abs().mean().item()
            total_mae += mae

        src_s = torch.randn(BATCH_SIZE, seq_len_src, EMBED_DIM, device=device)
        tgt_s = torch.randn(BATCH_SIZE, seq_len_tgt, EMBED_DIM, device=device)
        tgt_mask_s = torch.triu(torch.ones(seq_len_tgt, seq_len_tgt, dtype=torch.bool), diagonal=1).to(device)
        src_kpm_s = torch.zeros(BATCH_SIZE, seq_len_src, dtype=torch.bool).to(device)
        tgt_kpm_s = torch.zeros(BATCH_SIZE, seq_len_tgt, dtype=torch.bool).to(device)
        mem_kpm_s = src_kpm_s

        y_gnn_s = trans_gnn(
            src_s, tgt_s,
            src_mask=None,
            tgt_mask=tgt_mask_s,
            src_key_padding_mask=src_kpm_s,
            tgt_key_padding_mask=tgt_kpm_s,
            memory_key_padding_mask=mem_kpm_s
        )
        y_std_s = std_trans(
            src_s, tgt_s,
            src_mask=None,
            tgt_mask=tgt_mask_s,
            memory_mask=None,
            src_key_padding_mask=src_kpm_s,
            tgt_key_padding_mask=tgt_kpm_s,
            memory_key_padding_mask=mem_kpm_s
        )
        out_shape_gnn = tuple(y_gnn_s.shape)
        out_shape_std = tuple(y_std_s.shape)
        avg_mae = total_mae / NUM_ITERS

        print(f"Output shapes: GNN{out_shape_gnn}, Std{out_shape_std}")
        print(f"MAE 누적 합계: {total_mae:.6e}, 회당 평균 MAE: {avg_mae:.6e}")
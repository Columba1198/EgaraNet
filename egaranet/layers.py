"""
EgaraNet — Core layer definitions.

Modules:
    RMSNorm                   : Root Mean Square Layer Normalization
    SwiGLU                    : SwiGLU Feed-Forward Network
    TransposedAttentionTransformer: Cross-covariance (channel-wise) Transformer block
    AttentionPooling          : Learned-query attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps).
    More efficient than LayerNorm as it skips the mean-subtraction step.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (x_fp32 / rms) * self.weight.to(torch.float32)
        return out.to(orig_dtype)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Three-layer gated FFN: gate, up-projection, and down-projection.
    Hidden dimension is computed as round_up(floor(dim * 8/3), multiple).

    Args:
        dim: Input and output dimension.
        multiple: Hidden dimension is rounded up to the nearest multiple
                  of this value for hardware efficiency.
    """

    def __init__(self, dim: int, multiple: int = 64):
        super().__init__()
        hidden_dim = int(dim * 8 / 3)
        hidden_dim = ((hidden_dim + multiple - 1) // multiple) * multiple
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransposedAttentionTransformer(nn.Module):
    """Transposed Attention Transformer (TAT).

    A Transformer layer that computes attention in channel space rather than
    spatial space. Inspired by the observation that Gram matrices of feature
    maps encode style independent of content (Gatys et al., CVPR 2016).

    The key idea:
        1. Transpose Q: shape (N, HeadDim) → (HeadDim, N)
        2. Attention: (Q.T @ K) → (HeadDim, HeadDim) channel correlation matrix
        3. This replaces the standard (N, N) spatial attention with
           a channel-to-channel attention map

    This discards spatial/positional information, preserving only how features
    (channels) correlate with each other — the style signature.

    Extends the Transposed Attention from MANIQA (Yang et al., arXiv:2204.08958)
    into a complete Transformer block with RMSNorm and SwiGLU FFN.

    Args:
        dim: Model dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        eps: Epsilon for RMSNorm.
        swiglu_multiple: SwiGLU hidden dimension alignment.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-5,
                 swiglu_multiple: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-attention norm + QKV projection
        self.norm = RMSNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.proj = nn.Linear(dim, dim)

        # FFN
        self.norm_ffn = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLU(dim, multiple=swiglu_multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C] where N = number of spatial tokens.

        Returns:
            Output tensor of shape [B, N, C].
        """
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x)

        # QKV projection
        # [B, N, 3*C] -> [B, N, 3, Heads, HeadDim] -> [3, B, Heads, N, HeadDim]
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        # QK-Norm for training stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transposed attention: (HeadDim, N) @ (N, HeadDim) → (HeadDim, HeadDim)
        q = q.transpose(-2, -1)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        v = v.transpose(-2, -1)
        x = attn @ v  # (HeadDim, HeadDim) @ (HeadDim, N) → (HeadDim, N)

        # Restore shape: [B, Heads, HeadDim, N] → [B, N, Heads, HeadDim] → [B, N, C]
        x = x.transpose(-2, -1).reshape(B, N, C)
        x = self.proj(x)
        x = x + shortcut

        # Feed-forward
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AttentionPooling(nn.Module):
    """Learned-query attention pooling.

    Uses a single learnable query token to attend to the full sequence,
    collapsing [B, N, C] → [B, C].

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads in MultiheadAttention.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Pooled tensor of shape [B, C].
        """
        B = x.size(0)
        query = self.pool_query.expand(B, -1, -1)
        out, _ = self.pool_attn(query, x, x)
        return out.squeeze(1)

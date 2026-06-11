"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    import torch.nn.functional as F

    input_tensor, mask, weights, config = data
    hidden_dim = config['hidden_dim']

    # Fuse all 5 projections into one GEMM
    fused_w = torch.cat([
        weights['left_proj.weight'],
        weights['right_proj.weight'],
        weights['left_gate.weight'],
        weights['right_gate.weight'],
        weights['out_gate.weight'],
    ], dim=0)

    # LayerNorm input (float32)
    x = F.layer_norm(input_tensor, [input_tensor.shape[-1]],
                     weights['norm.weight'], weights['norm.bias'])

    # Single fused GEMM: (B, N, N, 5*H)
    proj_all = F.linear(x, fused_w)

    left_proj  = proj_all[..., :hidden_dim]
    right_proj = proj_all[..., hidden_dim:2*hidden_dim]
    left_gate  = proj_all[..., 2*hidden_dim:3*hidden_dim].sigmoid()
    right_gate = proj_all[..., 3*hidden_dim:4*hidden_dim].sigmoid()
    out_gate   = proj_all[..., 4*hidden_dim:5*hidden_dim].sigmoid()

    # Apply mask and gates (float32)
    mask_u = mask.unsqueeze(-1)
    left  = left_proj  * left_gate  * mask_u   # (B, N, N, H)
    right = right_proj * right_gate * mask_u   # (B, N, N, H)

    # Einsum: b i k h, b j k h -> b i j h  (contraction over k)
    # Use fp16 inputs for tensor core throughput; force fp32 accumulation
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    bs, seqlen, _, H = left.shape
    left_bh  = left.half().permute(0, 3, 1, 2).reshape(bs * H, seqlen, seqlen)
    right_bh = right.half().permute(0, 3, 1, 2).reshape(bs * H, seqlen, seqlen)
    out_bh = torch.bmm(left_bh, right_bh.transpose(-1, -2))  # fp16 in, fp32 acc
    out = out_bh.float().reshape(bs, H, seqlen, seqlen).permute(0, 2, 3, 1)

    # LayerNorm + out_gate + final projection (float32 throughout)
    out = F.layer_norm(out, [hidden_dim],
                       weights['to_out_norm.weight'], weights['to_out_norm.bias'])
    out = out * out_gate
    return F.linear(out, weights['to_out.weight'])

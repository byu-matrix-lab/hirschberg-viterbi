import torch
from torch import Tensor

# TODO: map to torchaudio signature

# TODO: Add wrapper functions and register fake ones here

# determines what gets imported by wild cart from this file
# __all__ = ["mymuladd", "myadd_out"]

def viterbi(log_probs: Tensor, target: Tensor, blank: int) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.hirschberg_viterbi.viterbi.default(log_probs, target, blank)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
# @torch.library.register_fake("hirschberg_viterbi::viterbi")
# def _(log_probs, target, blank):
#     # torch._check(a.shape == b.shape)
#     torch._check(log_probs.dtype == torch.float)
#     torch._check(target.dtype == torch.int)
#     torch._check(log_probs.device == target.device)
#     return torch.empty_like(log_probs)


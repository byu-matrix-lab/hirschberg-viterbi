import torch
from torch import Tensor

# determines what gets imported by wild cart from this file
__all__ = [
    "raw_viterbi",
    "raw_hirschberg_viterbi",
    "raw_pruned_viterbi",
    "raw_pruned_hirschberg_viterbi",
    "viterbi",
    "hirschberg_viterbi",
    "pruned_viterbi",
    "pruned_hirschberg_viterbi"
]

# TODO: current kernel does not support input lengths parameter for compilation
# TODO: rename conf to confidence

def raw_viterbi(log_probs: Tensor, targets: Tensor, blank: int = 0) -> Tensor:
    return torch.ops.hirschberg_viterbi.viterbi.default(log_probs, targets, blank)

def raw_hirschberg_viterbi(log_probs: Tensor, targets: Tensor, blank: int = 0, soft_mem_limit: int = 1000) -> Tensor:
    return torch.ops.hirschberg_viterbi.hirschberg_viterbi.default(log_probs, targets, blank, soft_mem_limit)

def raw_pruned_viterbi(log_probs: Tensor, targets: Tensor, blank: int = 0, var_rat: float = 3.7, conf: float = 0.99, accuracy: float = 0.97, precision: float = -1.0, recall: float = -1.0, padding: float = 5.0) -> Tensor:
    return torch.ops.hirschberg_viterbi.pruned_viterbi.default(log_probs, targets, blank, var_rat, conf, accuracy, precision, recall, padding)

def raw_pruned_hirschberg_viterbi(log_probs: Tensor, targets: Tensor, blank: int = 0, var_rat: float = 3.7, conf: float = 0.99, accuracy: float = 0.97, precision: float = -1.0, recall: float = -1.0, padding: float = 5.0, soft_mem_limit: int = 1000) -> Tensor:
    return torch.ops.hirschberg_viterbi.pruned_hirschberg_viterbi.default(log_probs, targets, blank, var_rat, conf, accuracy, precision, recall, padding, soft_mem_limit)

# TODO: maybe use typeddict to support auto complete
def torch_wrapper(log_probs, targets, input_lengths, target_lengths, blank, method, **kwargs):
    torch._check(log_probs.dim() == 3, lambda: "Log probs should have shape [batch size, sequence length, character set]")
    torch._check(targets.dim() == 2, lambda: "Targets should have shape [batch size, transcription length]")
    torch._check(log_probs.size(0)==1 and targets.size(0)==1, lambda: "Batch size must be one for this implementation")

    sliced_log_probs = log_probs[0]
    if input_lengths is not None:
        torch._check(input_lengths.dim() == 1, lambda: "Input lengths should have shape [batch size]")
        sliced_log_probs = sliced_log_probs[:input_lengths[0]]

    sliced_targets = targets[0]
    if target_lengths is not None:
        torch._check(target_lengths.dim() == 1, lambda: "Target lengths should have shape [batch size]")
        sliced_targets = sliced_targets[:target_lengths[0]]

    alignment = method.default(sliced_log_probs, sliced_targets, **kwargs)

    # -1 handles final blank > len sliced_targets
    labels = torch.where((alignment%2)==0, blank, sliced_targets[(alignment-1)//2])
    confs = sliced_log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    return labels.unsqueeze(0), confs.unsqueeze(0)

def viterbi(log_probs: Tensor, targets: Tensor, input_lengths: Tensor | None = None, target_lengths: Tensor | None = None, blank: int = 0, **kwargs) -> Tensor:
    return torch_wrapper(log_probs, targets, input_lengths, target_lengths, blank, torch.ops.hirschberg_viterbi.viterbi, **kwargs)

def hirschberg_viterbi(log_probs: Tensor, targets: Tensor, input_lengths: Tensor | None = None, target_lengths: Tensor | None = None, blank: int = 0, **kwargs) -> Tensor:
    return torch_wrapper(log_probs, targets, input_lengths, target_lengths, blank, torch.ops.hirschberg_viterbi.hirschberg_viterbi, **kwargs)

def pruned_viterbi(log_probs: Tensor, targets: Tensor, input_lengths: Tensor | None = None, target_lengths: Tensor | None = None, blank: int = 0, **kwargs) -> Tensor:
    return torch_wrapper(log_probs, targets, input_lengths, target_lengths, blank, torch.ops.hirschberg_viterbi.pruned_viterbi, **kwargs)

def pruned_hirschberg_viterbi(log_probs: Tensor, targets: Tensor, input_lengths: Tensor | None = None, target_lengths: Tensor | None = None, blank: int = 0, **kwargs) -> Tensor:
    return torch_wrapper(log_probs, targets, input_lengths, target_lengths, blank, torch.ops.hirschberg_viterbi.pruned_hirschberg_viterbi, **kwargs)

# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
def fake_checker(log_probs, targets, **kwargs):
    torch._check(log_probs.dim() == 2)
    torch._check(targets.dim() == 1)
    torch._check(log_probs.dtype == torch.float or log_probs.dtype == torch.double)
    torch._check(targets.dtype == torch.int)
    torch._check(log_probs.device == targets.device)
    return torch.empty((log_probs.size(0),), dtype=targets.dtype, device=log_probs.device)

torch.library.register_fake("hirschberg_viterbi::viterbi", fake_checker)
torch.library.register_fake("hirschberg_viterbi::hirschberg_viterbi", fake_checker)
torch.library.register_fake("hirschberg_viterbi::pruned_viterbi", fake_checker)
torch.library.register_fake("hirschberg_viterbi::pruned_hirschberg_viterbi", fake_checker)

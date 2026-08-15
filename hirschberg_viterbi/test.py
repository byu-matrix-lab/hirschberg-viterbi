print('starting test.py')

import torch
import torchaudio
import time
import hirschberg_viterbi

torch.manual_seed(0)

method = 'test'
device_str = 'cpu'
device = torch.device(device_str)

seconds = 1200 #200
charset = 154
log_probs = torch.rand((int(50*seconds), charset), device=device).log_softmax(dim=-1) # predictions predict 50 characters per second
cleaned = torch.randint(low=1, high=charset, size=(int(12.7 * seconds),), device=device) # On average conference talks have 12.7 characters per second

log_probs = log_probs.float()
cleaned = cleaned.int()

start = time.time()

temp1 = torch.ops.hirschberg_viterbi.hirschberg_viterbi(log_probs, cleaned)

# print()
# print()

# print(log_probs.flags)

# print(log_probs.dtype)
# print(cleaned.dtype)

# temp1 = hirschberg_viterbi_impl.pruned_hirschberg_viterbi(log_probs.numpy(), cleaned.numpy(), soft_mem_limit=1000)
# temp1 = hirschberg_viterbi_impl.pruned_viterbi(log_probs.numpy(), cleaned.numpy())
# temp1 = hirschberg_viterbi_impl.hirschberg_viterbi(log_probs.numpy(), cleaned.numpy(), soft_mem_limit=1000)
# temp1 = hirschberg_viterbi_impl.viterbi(log_probs.numpy(), cleaned.numpy())

# temp = torchaudio.functional.forced_align(log_probs.unsqueeze(0), cleaned.unsqueeze(0), T_batch, U_batch)
end = time.time()

print(end - start)

start = time.time()
a, b = torchaudio.functional.forced_align(log_probs.unsqueeze(0), cleaned.unsqueeze(0))
end = time.time()
print(end - start)

temp1 = temp1.masked_fill((temp1%2)==0, 0)
temp1 = torch.where(temp1 != 0, cleaned[temp1//2], temp1)
# print((a[0] == temp1))
print((a[0] == temp1).float().mean().item())


a2, b = torchaudio.functional.forced_align(log_probs.unsqueeze(0).double(), cleaned.unsqueeze(0))

# print((a2[0] == temp1))
print((a2[0] == temp1).float().mean().item())

print((a == a2).float().mean().item())
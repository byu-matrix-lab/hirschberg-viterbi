import torch
import torchaudio
import time
import hirschberg_viterbi

torch.manual_seed(0)

method = 'test'
device_str = 'cpu'
device = torch.device(device_str)

seconds = 20 #200
charset = 154
log_probs = torch.rand((int(50*seconds), charset), device=device).log_softmax(dim=-1) # predictions predict 50 characters per second
cleaned = torch.randint(low=1, high=charset, size=(int(12.7 * seconds),), device=device) # On average conference talks have 12.7 characters per second

log_probs = log_probs.float()
cleaned = cleaned.int()

start = time.time()

my_align, my_confs = hirschberg_viterbi.viterbi(log_probs.unsqueeze(0), cleaned.unsqueeze(0), blank=0)



end = time.time()

print('Pruned Hirschberg-Viterbi runtime (s):', end - start)

# start = time.time()
# a, b = torchaudio.functional.forced_align(log_probs.unsqueeze(0), cleaned.unsqueeze(0))
# end = time.time()
# print(end - start)

start = time.time()
a2, b = torchaudio.functional.forced_align(log_probs.unsqueeze(0).double(), cleaned.unsqueeze(0))
end = time.time()
print('Torchaudio runtime (s):', end - start)

# print((a[0] == temp1))
# print((a[0] == temp1).float().mean().item())

# print((a2[0] == temp1))
# print((a2[0] == temp1).float().mean().item())
print('Outputs Match:', (a2 == my_align).all().item() and (b == my_confs).all().item())

# print((a == a2).float().mean().item())
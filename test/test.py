import torch
import torchaudio
import time
import hirschberg_viterbi

torch.manual_seed(0)

method = 'test'
device_str = 'cpu'
device = torch.device(device_str)

seconds = 1200
charset = 154
log_probs = torch.rand((int(50*seconds), charset), device=device).log_softmax(dim=-1).contiguous() # predictions predict 50 characters per second
cleaned = torch.randint(low=1, high=charset, size=(int(12.7 * seconds),), device=device).contiguous() # On average conference talks have 12.7 characters per second

log_probs = log_probs.float()
cleaned = cleaned.int()

# warp up the code, initial module load has around 140 ms latency
# didn't see this latency with torchaudio, but included for fair comparison
hirschberg_viterbi.hirschberg_viterbi(log_probs[:100].unsqueeze(0), cleaned[:10].unsqueeze(0))
torchaudio.functional.forced_align(log_probs[:100].unsqueeze(0), cleaned[:10].unsqueeze(0))

start = time.time()

my_align, my_confs = hirschberg_viterbi.hirschberg_viterbi(log_probs.unsqueeze(0), cleaned.unsqueeze(0))

end = time.time()

print('Hirschberg-Viterbi runtime (s):', end - start)

start = time.time()
a2, b = torchaudio.functional.forced_align(log_probs.unsqueeze(0).double(), cleaned.unsqueeze(0))
end = time.time()
print('Torchaudio runtime (s):', end - start)

print('Outputs Match:', (a2 == my_align).all().item(), (b == my_confs).all().item())

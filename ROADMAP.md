# Roadmap

I don't have a set timeline for these, but these are some of the main things I would like to work on for this project.

## Unit Tests

Just so I don't break things in a future release.

## Portability

I have done all of the development and testing for this system on Linux. I expect that building it on Windows/macOS will need some configuration changes.

## VAD

I have experimented with two methods for handling silence in audio: removing VAD-tagged silence before running forced alignment and lowering the recall parameter to allow for higher worst-case bounds while leaving the silence present.

Both methods have issues though, as fully removing the silence introduces cascaded errors, especially when the input audio is noisy. On the other hand, lowering the recall parameter maintains accuracy but significantly widens the pruning space when a lot of the audio is silence.

I am interested in passing in a tensor of VAD probabilities per `log_probs` timestep. My current pruning prior scales with the length of the audio file. I would be interested in taking a prefix sum of the VAD probabilities to find the expected speech duration in the file. The pruning bounds could be centered on this prefix sum before mapping to timesteps.

I think this would guide the bounds to follow the speech in the file while avoiding cascaded errors, but I would want to benchmark its pruning accuracy compared to the current methods.

## CUDA

The current implementation is CPU-only, and adding CUDA support could greatly improve runtime.

The current `torchaudio` [CUDA implementation](https://github.com/pytorch/audio/blob/fbf1d75b2a2cc947b1ae23aed6ff229cce540856/src/libtorchaudio/forced_align/gpu/compute.cu) runs on a single SM. It launches a separate kernel for each timestep in the input, and iteratively copies the $\mathcal{O}(n^2)$ backtracking pointers out to host memory, which creates a lot of synchronization overhead.

The [`NeMo forced aligner`](https://github.com/NVIDIA-NeMo/Speech/blob/00278b0bd95bb2b8174b88012aa21b003c59d2e9/nemo/collections/asr/parts/utils/aligner_utils.py#L726) supports running on CUDA, but it also has quadratic memory usage and is written in normal PyTorch, leaving more space for optimizations.

The CUDA kernel design that I have in mind would make use of the bidirectional and recursive nature of the Hirschberg--Viterbi algorithm.

The CPU implementation currently recurses until the subproblem fits within 1000 bytes. It is difficult to predict exactly when this will happen, but the memory usage has a strict upper bound based on the number of timesteps in the problem, as each timestep can only advance the alignment by one character (two CTC states). For the CUDA implementation, I am considering having the recursion stop once the number of timesteps in each subproblem is around 30 (subject to benchmarking to optimize). This allows the recursion depth to be known at runtime, just by looking at the length of the sequence.

I am planning for each kernel launch to take a list of subproblems at one recursion depth and solve them while generating the list for the next recursion depth. This will require only $\mathcal{O}(\log n)$ kernel launches. Also, each subproblem can be solved independently as separate thread blocks, allowing for parallelization over SMs.

When the number of subproblems is less than the number of available SMs, the forward and backwards passes could be further split across thread blocks to allow even greater parallelization. This would be particularly useful when solving the top-level problem.

Since in-kernel memory allocations are relatively expensive, the host could request a single working memory pool for all thread blocks at the start of the alignment. A separate kernel launch could reshard this memory pool for the current subproblem sizes at each level of recursion using a prefix sum.

The bounds themselves could also be computed on device using prefix min/max operations, fully eliminating inter-device communication during alignment. This could enable CUDA graph captures for alignments, though the API would need to change to accept sequence length parameters. I expect that graph captures would probably have limited benefit though, as I estimate that aligning a 3-hour audio would only require about 40 kernel launches under this setup, which is relatively small compared to the amount of work that will be done. The current `torchaudio` implementation launches $540,000$ kernels to solve the same problem.

I think the same would also be possible with the potential VAD input described above, though I have yet to design how the binding between it and the prior would work efficiently in parallel.

# Hirschberg-Viterbi

Optimized forced alignment with linear memory.

## Optimizations

This code is intended to be an optimized substitute for [`torchaudio.functional.forced_align`](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html), especially for long input sequences.

We introduce optimizations over torchaudio's implementation:

1. We use the Hirschberg algorithm to perform alignments in place, which has $\mathcal{O}(n)$ memory usage with a very small constant factor. Torchaudio creates a $\mathcal{O}(n^2)$ backtracking matrix for each alignment, which requires significant memory movement for large inputs, especially when processing on CUDA, as this data is stored in host memory.

2. We treat speech duration as a random walk, allowing us to estimate bounds for an alignment with the normal distribution. We use these bounds to prune the search space.

We also give credit to [`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner), for helping inspire this work. `ctc-forced-aligner` also noted issues with torchaudio's memory usage, and addressed them by compressing backtracking data into 2 bits instead of a full byte.

## Installation

Install from source:

```
git clone https://github.com/byu-matrix-lab/hirschberg-viterbi
cd hirschberg_viterbi
pip install -e .
```

Use `USE_CUDA=0` to create a cpu-only build on CUDA systems.

## Usage

There are 4 alignment methods in this package: viterbi, hirschberg_viterbi, pruned_viterbi, and pruned_hirschberg_viterbi, depending which optimizations you want to use. We recommend always using the Hirschberg optimization, as it was always beneficial in our experiments.

The pruned_* implementations prune the alignment search space, and have a chance to produce inaccurate alignments, especially for audios with large changes in speaking rate. This method requires you to give a lower bound for the accuracy of the transcription that you are aligning (more search space is pruned the more accurate the transcription is).

We make our function signatures similar to [`torchaudio.functional.forced_align`](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html) for each of adoption. 

In our implementation, the dtype of `log_probs` can be either `double` or `float` during compute, but the `targets` input must be `int` and not `long`.

Each optimization adds some number of extra parameters.

- Hirschberg optimization

    - `soft_mem_limit` - Stop recursion early when the size of the backtracking trellis does not exceed this size in bytes. Defaults to 1000 B.

- Pruning optimization

    - `accuracy` - This is the main parameter you should change when using this method. It needs to be a *lower bound* for the accuracy of the transcription. The pruning space is increased based on the worst-case possibilities for eronuous transcript. This defaults to 0.97.
    - `precision`, `recall` - These let you break the transcription accuracy into precision and recall, which indicate how much of the transcript does not have audio, and how much of the audio does not have a transcript. These should also be lower bounds. These default to -1 to indicate that the shared accuracy parameter is being used instead.
    - `confidence` - The confidence level for the confidence window used to produce pruning bounds. Defaults to 0.99.
    - `var_rat` - This parameter should be an upper bound on the ratio between the variance and mean of the durations of graphemes, and is used in scaling the expected duration of the text to match the audio length. Defaults to 3.7 based on our analysis. Probably don't change this unless you are doing research.
    - `padding` - This parameter sets a lower bound for the timesteps in each half of the window around the diagonal. This is needed for the edges of the audio, where the pruning model breaks down because the sequence is not long enough for normality to form. This defaults to 375 (7.5 seconds on each side * 50 timesteps per second), to create a minimum of a 15 second window centered on the diagonal. I called it padding because the pruning window is padded on the ends up to this size, but it could probably have a better name.

We will be adding a silence parameter soon to indicate an upper bound for the amount of silence in the audio. For now, you can subtract silence from the recall parameter, as they both indicate part of the audio that does not have a transcription.

For now, you must move the inputs to the CPU for computation. I have a loose design for a CUDA kernel planned with much higher concurrency, that I will implement it if there is sufficient demand / use cases beyond the CPU implementation provided.

## Output

Each function matches the return signature of `torchaudio.functional.forced_align` in returning the character at each timestep and the model confidence for those characters.

Each method also has a `raw_*` equivalent, such as `raw_hirschberg_viterbi`. These return the direct indices of the CTC alignment at each timestep. Odd indicies represent locations in the transcript (when floor divided by 2), while even indices are the inserted CTC blanks between characters. This output was more useful for our usecases, and the non-raw functions are just python wrappers that convert it to match the torchaudio output. Note that this output is also int32. The input for the `raw_*` functions are slightly different, in that they take unbatched input without the `input_lengths` and `target_lengths` parameters.

## Citation

```bibtex
citation here
```

# Adapted from:
# https://github.com/pytorch/extension-cpp/blob/master/extension_cpp_stable/setup.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "hirschberg_viterbi"

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "0") == "1" # Default is no cuda for now
    if debug_mode:
        print("Compiling in debug mode")

    if use_cuda and CUDA_HOME is None:
        raise RuntimeError("USE_CUDA=1, but no CUDA toolkit was found")

    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
            # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
            # stable API subset from torch
            # Format: [MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]
            # 2.10.0 = 0x020A000000000000
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            # NVCC also needs TORCH_TARGET_VERSION for stable ABI in CUDA code
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            # USE_CUDA is currently needed for aoti_torch_get_current_cuda_stream
            # declaration in shim.h. This will be improved in a future release.
            "-DUSE_CUDA",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.relpath(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = sorted(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = sorted(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=True,
        )
    ]

    return ext_modules


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
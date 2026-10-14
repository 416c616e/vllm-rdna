# MonoKernel 2a — FFN kernel, standalone M1 bench

`csrc/rocm/mono_ffn_rdna2.cu` is built here as a JIT torch extension (it is not
wired into CMake / `torch_bindings.cpp`; M2 was not started). Results:
[`docs/rdna/monokernel/2a-m1-results.md`](../../docs/rdna/monokernel/2a-m1-results.md).

    GPU=2 bash tools/monokernel-2a/m1_final.sh          # widths 3/6/12/24, prefetch off/on
    GPU=2 bash tools/monokernel-2a/run_m1.sh --widths 3 --layers 4   # quick check

* `m1_bench.py` — correctness against today's op sequence and an fp32 golden,
  and hipGraph timing over `--layers` distinct layers (cold Infinity Cache).
* `ext_binding.cpp` — pybind entry points (`mono_ffn`, `workspace_bytes`,
  `error`, `phase_ts`).
* `MONO_FFN_PREFETCH=1` enables barrier-time prefetch, `MONO_FFN_VERBOSE=1`
  prints the residency cap, `MONO_MIN_BLOCKS=2` builds the 2-WGs-per-CU variant.

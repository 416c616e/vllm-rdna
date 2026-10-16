# RDNA2 MonoKernel tools (`--monokernel`)

The model code is `vllm/models/qwen4_exp/amd/mono_decode_rdna2.py` +
`mono/rdna2/`; the pytest suite is `tests/models/qwen4_exp/test_amd_mono_rdna2*.py`.
History and results: [`docs/rdna/monokernel/`](../../../docs/rdna/monokernel/).

Serve drivers (HIP 6-9 lock, `serve_validate.sh` gates):

    ARM=on|off|env GPUS=6,7,8,9 PORT=18790 TAG=x RECIPE=flashnext-mtp0 \
        bash tools/rdna/mono/serve_2a.sh          # one serve; ARM=on adds --monokernel
    SPECS='mtp0|off|a|;mtp0|on|a|' CELLS="1024:512:1 1024:512:8" GPUS=6,7,8,9 \
        bash tools/rdna/mono/run_specs.sh         # a queue of serves, one lock each

`MONO_CFG='"k2_max_rows":8'` adds `rocm_mono_decode` knobs; `ARM=on-t|off-t`
turn on the in-serve timing (mono vs today's modules under the same marks).

Standalone checks and benches (`PYTHONPATH=<tree>`):

* `check_op.py` — registered `mono_k2_rdna2` vs today's ops, widths 1..24,
  graph replay, fail-stop NaN, stats marks.
* `k1_bench.py`, `m1_bench.py`, `gdn_fused_bench.py`, `qsa_bench.py` —
  correctness and graph timing of K1, K2, GDN, QSA against today's op
  sequences (JIT builds of the `csrc/rocm/mono_*_rdna2.cu` sources).
* `qsa_normrope_check.py`, `qsa_store3_check.py` — QSA fused norm+RoPE and the
  one-launch indexer stores.
* `m1_final.sh`, `run_m1.sh`, `sweep.sh` — the M1 standalone runs;
  `MONO_FFN_PREFETCH=1`, `MONO_FFN_VERBOSE=1`, `MONO_MIN_BLOCKS=2` build variants.

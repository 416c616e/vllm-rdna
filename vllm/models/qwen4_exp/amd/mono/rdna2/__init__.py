# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MonoKernel on gfx1030 (RDNA2): the parts behind ``--monokernel``.

``k1`` attention-side HC preamble, ``k2`` persistent FFN block, ``gdn`` fused
GDN decode core, ``qsa`` QSA decode; ``layout`` holds the shared weight views,
workspace and all-reduce attach, ``debug`` the in-serve timing. The switch,
the deployment checks and the hook installation are in
``vllm/models/qwen4_exp/amd/mono_decode_rdna2.py``; see README.md.
"""

from dataclasses import dataclass, fields


@dataclass
class Options:
    """``--additional-config '{"rocm_mono_decode": {...}}'`` (only read with
    ``--monokernel``). The defaults are the validated all-on configuration."""

    k1: bool = True  # attention-side HC preamble on the persistent kernel
    k1_max_rows: int = 24
    k2: bool = True  # decode FFN block in one persistent launch
    k2_max_rows: int = 8  # width gate: K2 up to this many rows, modules above
    k2_modes: str = "full,piecewise,eager"
    k2_num_wg: int = 0  # 0: residency cap
    k2_sleep: int = 0  # tuning: barrier poll backoff
    k2_flags: int = 0  # tuning flags (workspace tail)
    k2_ar: bool = False  # FFN all-reduce inside K2 on the rdna_ar buffers
    gdn: bool = True  # fused GDN decode core
    gdn_max_rows: int = 64
    gdn_graph: bool = False  # record the fused GDN core in FULL graphs
    qsa: bool = True  # QSA decode: bounded indexer, HIP sparse decode, fusions
    timing: bool = False  # in-serve timing marks / per-width stats (debug)
    modules: bool = False  # timing reference: today's modules under the marks
    hw_queues: int = 4  # GPU_MAX_HW_QUEUES floor for the workers (0: leave)

    @classmethod
    def from_dict(cls, extra: dict) -> "Options":
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(extra) - set(known))
        if unknown:
            raise ValueError(
                f"--monokernel: unknown rocm_mono_decode keys {unknown} "
                f"(known: {sorted(known)})"
            )
        opts = cls()
        for k, v in extra.items():
            cur = getattr(opts, k)
            setattr(opts, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
        return opts

    def assign(self, other: "Options") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(other, f.name))


# The active options of this process (set by mono_decode_rdna2.install).
OPTS = Options()

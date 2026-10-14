# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel 2a (gfx1030): one persistent launch for a decoder layer's FFN block.

Opt-in with ``--additional-config '{"rdna_mono_ffn": true}'`` (config, not an
env var, so it reaches every worker). It covers what runs between the
attention all-reduce and the FFN all-reduce: the MLP hyper-connection combine
+ mix, router (fp32 logits and top-k), shared expert and the routed symmetric
W4A16 experts (``csrc/rocm/mono_ffn_rdna2.cu``). The all-reduce stays outside.

The FFN block is per token, so the kernel is valid for any batch of at most
``rdna_mono_ffn_max_rows`` rows (<= 24): FULL and PIECEWISE CUDA graph
captures and eager steps alike (``rdna_mono_ffn_modes``, default all three).
Larger batches (prefill chunks) and unsupported layers run today's modules;
the MTP drafter never takes it. The decision is made at run time inside one
opaque custom op, so torch.compile sees a single op and the branch follows
the actual batch, not the trace-time size.

Fail-stop: a timed-out grid barrier leaves a sticky record; later launches
exit at once and fill the FFN output with NaN instead of returning stale data.

Diagnostics (additional_config, off by default): ``rdna_mono_ffn_timing``
brackets the FFN block + its all-reduce inside captured graphs with timestamp
kernels and turns on the kernel's per-width phase/entry stats; the per-width
means are logged on eager steps. ``rdna_mono_ffn_modules`` keeps today's
modules everywhere (the switch-off reference with the same marks).
"""

import time

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MAX_ROWS = 24
_WORKSPACES: dict[int, torch.Tensor] = {}
_STATS: dict[int, torch.Tensor] = {}
_LOGGED: set = set()
_TIMING = False
_FORCE_MODULES = False
_MAX_ROWS = 8  # width gate: mono for <= this many rows, today's modules above
_NUM_WG = 0  # 0: residency cap
_TUNE = {"sleep": 0, "flags": 0}
_MODES = {"full", "piecewise", "eager"}
_STAT_COLS = 24
_STAT_ROWS = MAX_ROWS + 1
_STAT_WORDS = (_STAT_ROWS + 1) * _STAT_COLS
_TICKS_PER_US = 100.0  # wall_clock64 runs at 100 MHz on V620
_last_log = [0.0]


def mono_ffn_requested(vllm_config) -> bool:
    global _TIMING, _FORCE_MODULES
    extra = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(extra, dict):
        return False
    global _MAX_ROWS, _NUM_WG
    _TIMING = bool(extra.get("rdna_mono_ffn_timing", False))
    _FORCE_MODULES = bool(extra.get("rdna_mono_ffn_modules", False))
    _MAX_ROWS = min(MAX_ROWS, int(extra.get("rdna_mono_ffn_max_rows", 8)))
    _NUM_WG = int(extra.get("rdna_mono_ffn_num_wg", 0))
    _TUNE["sleep"] = int(extra.get("rdna_mono_ffn_sleep", 0))
    _TUNE["flags"] = int(extra.get("rdna_mono_ffn_flags", 0))
    modes = extra.get("rdna_mono_ffn_modes", "full,piecewise,eager")
    _MODES.clear()
    _MODES.update(x.strip() for x in str(modes).split(",") if x.strip())
    return bool(extra.get("rdna_mono_ffn", False))


def mono_ffn_supported(vllm_config) -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        return False
    import vllm._custom_ops  # noqa: F401  (loads _rocm_C)

    return hasattr(torch.ops._rocm_C, "mono_ffn_rdna2")


def _workspace(device: torch.device) -> torch.Tensor:
    """One zero-initialized workspace per device, allocated outside capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACES.get(idx)
    if ws is None:
        nbytes = int(torch.ops._rocm_C.mono_ffn_rdna2_workspace_bytes(2560, 320, 336, 512, 160, 640))
        ws = torch.zeros(nbytes + 8192, dtype=torch.uint8, device=device)
        _WORKSPACES[idx] = ws
        # stats / tuning block right after the kernel's workspace
        st = ws[nbytes : nbytes + _STAT_WORDS * 8].view(torch.int64)
        sc = _STAT_ROWS * _STAT_COLS
        st[sc + 0] = -1  # entry_min
        st[sc + 5] = _TUNE["sleep"]
        st[sc + 6] = _TUNE["flags"]
        if _TIMING:
            st[sc + 3] = 1  # enabled
            _STATS[idx] = st
    return ws


def _log_stats():
    """Per-width means (µs) of the serve timing, all devices of this process."""
    now = time.monotonic()
    if now - _last_log[0] < 30.0:
        return
    _last_log[0] = now
    for idx, st in _STATS.items():
        a = st.view(-1, _STAT_COLS).cpu()
        for w in range(1, _STAT_ROWS):
            r = a[w].tolist()
            n, nm, ng = r[0], r[12], r[16]
            if n == 0 and nm == 0:
                continue
            us = lambda v, k: v / k / _TICKS_PER_US if k else 0.0  # noqa: E731
            ph = "/".join(f"{us(r[i], n):.1f}" for i in range(1, 8))
            logger.info(
                "rdna_mono_ffn timing dev%d w%d: region n=%d %.1f us | kernel n=%d "
                "total %.1f phases %s | entry spread %.1f wg0 %.1f | mark->first %.1f "
                "mark->last %.1f | rest n=%d %.1f us | attn n=%d %.1f us (hc %.1f)",
                idx, w, nm, us(r[13], nm), n, us(r[9], n), ph, us(r[8], n),
                us(r[10], n), us(r[14], n), us(r[15], n), ng, us(r[17], ng),
                r[18], us(r[19], r[18]), us(r[21], r[20]),
            )


class MonoFFN:
    """Per-layer view of the weights the kernel reads; None if unsupported."""

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.first = getattr(layer, "layer_idx", None) == 0
        self.ready = False
        self.reason = ""

    def setup(self) -> bool:
        """Resolve tensors after process_weights_after_loading (first call)."""
        if self.ready or self.reason:
            return self.ready
        try:
            self._setup()
            self.ready = True
        except Exception as e:  # unsupported geometry/quant: stay on modules
            self.reason = str(e)
            logger.warning_once(
                "rdna_mono_ffn disabled for %s: %s", type(self.layer).__name__, e
            )
        return self.ready

    def _setup(self):
        layer = self.layer
        hc = layer.mlp_hyper_connection
        mlp = layer.mlp
        for name in ("gate", "experts"):
            if not hasattr(mlp, name):
                raise ValueError(f"mlp has no {name}")
        if hc.hc_count != 4 or not hc.use_combine:
            raise ValueError("hyper-connection must be hc_count 4 with combine")
        down = hc.input_mix_weight_down_block_inject
        up = hc.input_mix_weight_up
        if getattr(down, "weight_i8", None) is not None or getattr(up, "weight_i8", None) is not None:
            raise ValueError("int8 HC shadows are not supported")
        experts = getattr(mlp.experts, "routed_experts", mlp.experts)
        qm = getattr(experts, "quant_method", None) or getattr(mlp.experts, "_quant_method", None)
        if type(qm).__name__ != "CompressedTensorsWNA16RDNA2MoEMethod":
            raise ValueError(f"routed experts use {type(qm).__name__}")
        if getattr(qm, "_zero_v2", False) or getattr(qm, "_w4a8", False):
            raise ValueError("asymmetric (v2 zeros) or W4A8 experts")
        w13 = experts.w13_weight_packed
        if w13.shape[0] != 128 or w13.shape[2] != 1280 or w13.shape[1] * 8 != 2560:
            raise ValueError(f"unexpected routed expert shape {tuple(w13.shape)}")
        se = getattr(mlp, "shared_expert", None)
        if se is None or getattr(se, "expert_gate", None) is None:
            raise ValueError("shared expert with expert gate required")
        gu = se.gate_up_proj.weight
        dn = se.down_proj.weight
        wg = se.expert_gate.weight
        if gu.dtype != torch.float16 or dn.dtype != torch.float16 or gu.shape[0] > 320:
            raise ValueError("shared expert must be fp16 with <= 160 local rows")
        router = mlp.gate.weight
        if router.dtype != torch.float16 or router.shape != (512, 2560):
            raise ValueError("router must be fp16 [512, 2560]")
        emap = getattr(experts, "expert_map", None)
        if emap is None:
            raise ValueError("expert parallelism (expert_map) required")
        loc = (emap >= 0).nonzero().flatten()
        base = int(loc[0]) if loc.numel() else -1
        if loc.numel() != 128 or not torch.equal(
            emap[base : base + 128].cpu(), torch.arange(128, dtype=emap.dtype)
        ):
            raise ValueError("local experts must be one contiguous block of 128")
        if getattr(experts, "top_k", 10) != 10 or not getattr(experts, "renormalize", True):
            raise ValueError("top-10 with renormalization required")
        self.t = dict(
            norm_w=hc.hc_norm.weight, wd=down.weight, wu=up.weight, wr=router,
            se_w1=gu, se_w2=dn, se_wg=wg.reshape(-1).contiguous(),
            w13=w13, s13=experts.w13_weight_scale, z13=experts.w13_qzeros,
            w2=experts.w2_weight_packed, s2=experts.w2_weight_scale,
            z2=experts.w2_qzeros,
        )
        self.base = base
        self.eps = float(hc.config.rms_norm_eps)
        self.ws = _workspace(router.device)

    def run(self, hidden_states, attn_out, injection):
        t = self.t
        m = hidden_states.shape[0]
        res_out = torch.empty_like(hidden_states)
        inj_out = torch.empty((m, 4), dtype=hidden_states.dtype, device=hidden_states.device)
        out = torch.empty_like(attn_out)
        torch.ops._rocm_C.mono_ffn_rdna2(
            hidden_states, attn_out, injection.contiguous(), t["norm_w"], t["wd"],
            t["wu"], t["wr"], t["se_w1"], t["se_w2"], t["se_wg"], t["w13"], t["s13"],
            t["z13"], t["w2"], t["s2"], t["z2"], res_out, inj_out, out, self.ws,
            self.base, 1, _NUM_WG, self.eps, 2000.0,
        )
        return res_out, out, inj_out


def timing_enabled() -> bool:
    return _TIMING


def _mark(kind: int, x: torch.Tensor) -> None:
    """Serve timing mark (see mono_mark_kernel); FULL-graph capture only."""
    m = x.shape[0]
    st = _STATS.get(x.device.index)
    if st is not None and 0 < m <= MAX_ROWS and _mode() != "eager":
        torch.ops._rocm_C.mono_ffn_rdna2_mark(st, kind, m)


def _mono_attn_hc(
    hidden_states: torch.Tensor,
    prev_out: torch.Tensor | None,
    prev_inj: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Timing only: today's attention-side HC call between marks 3 and 4."""
    from vllm.forward_context import get_forward_context

    hc = get_forward_context().no_compile_layers[layer_name].layer.attn_hyper_connection
    _mark(3, hidden_states)
    if prev_out is not None and prev_inj is not None:
        out = hc.combine_and_mix(hidden_states, prev_out, prev_inj)
    else:
        out = hc.mix(hidden_states)
    _mark(4, hidden_states)
    # a custom op may not return its inputs (mix() hands back hidden_states)
    ins = [t for t in (hidden_states, prev_out, prev_inj) if t is not None]
    return tuple(
        o.clone() if any(o.data_ptr() == t.data_ptr() for t in ins) else o for o in out
    )


def _mono_attn_hc_fake(hidden_states, prev_out, prev_inj, layer_name):
    m = hidden_states.shape[0]
    return (
        torch.empty_like(hidden_states),
        hidden_states.new_empty((m, hidden_states.shape[1] // 4)),
        hidden_states.new_empty((m, 4)),
    )


def _mode() -> str:
    """"full", "piecewise" or "eager" for the current call.

    Model Runner V2 captures FULL graphs by running the model with
    cudagraph_runtime_mode NONE inside torch.cuda.graph; PIECEWISE capture runs
    with mode PIECEWISE. Graph replays do not re-enter Python, so the
    capture-time choice is what replays."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not torch.cuda.is_current_stream_capturing():
        return "eager"
    if (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    ):
        return "piecewise"
    return "full"


def _mono_ffn_block(
    hidden_states: torch.Tensor,
    attn_out: torch.Tensor,
    injection: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context

    mono = get_forward_context().no_compile_layers[layer_name]
    layer = mono.layer
    m = hidden_states.shape[0]
    if not torch.cuda.is_current_stream_capturing():
        # Resolve weights / allocate the workspace on an eager step (profile
        # and warmup runs precede capture); nothing here may run in a capture.
        mono.setup()
    mode = _mode()
    fits = 0 < m <= MAX_ROWS and hidden_states.dtype == torch.float16 and mono.ready
    use = fits and not _FORCE_MODULES and m <= _MAX_ROWS and mode in _MODES
    key = (m, mode, use)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "rdna_mono_ffn: %d rows (%s) -> %s", m, mode, "mono kernel" if use else "modules"
        )
    timed = _TIMING and fits and mode != "eager"
    st = _STATS.get(hidden_states.device.index) if timed else None
    if _TIMING and not torch.cuda.is_current_stream_capturing():
        _log_stats()
    if st is not None:
        torch.ops._rocm_C.mono_ffn_rdna2_mark(st, 5, m)  # attention block done
        torch.ops._rocm_C.mono_ffn_rdna2_mark(st, 0 if mono.first else 2, m)
    if use:
        from vllm.distributed import tensor_model_parallel_all_reduce

        res_out, partial, inj_out = mono.run(hidden_states, attn_out, injection)
        result = res_out, tensor_model_parallel_all_reduce(partial), inj_out
    else:
        hs, block_input, inj = layer.mlp_hyper_connection.combine_and_mix(
            hidden_states, attn_out, injection
        )
        result = hs, layer.mlp(block_input), inj
    if st is not None:
        torch.ops._rocm_C.mono_ffn_rdna2_mark(st, 1, m)
    return result


def _mono_ffn_block_fake(hidden_states, attn_out, injection, layer_name):
    m = hidden_states.shape[0]
    return (
        torch.empty_like(hidden_states),
        torch.empty_like(attn_out),
        hidden_states.new_empty((m, 4)),
    )


direct_register_custom_op(
    op_name="rdna_mono_ffn_block",
    op_func=_mono_ffn_block,
    fake_impl=_mono_ffn_block_fake,
)
direct_register_custom_op(
    op_name="rdna_mono_attn_hc",
    op_func=_mono_attn_hc,
    fake_impl=_mono_attn_hc_fake,
)

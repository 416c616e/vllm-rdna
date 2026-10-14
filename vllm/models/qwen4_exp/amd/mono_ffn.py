# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel 2a (gfx1030): one persistent launch for a decoder layer's FFN block.

Opt-in with ``--additional-config '{"rdna_mono_ffn": true}'`` (config, not an
env var, so it reaches every worker). It covers what runs between the
attention all-reduce and the FFN all-reduce: the MLP hyper-connection combine
+ mix, router (fp32 logits and top-k), shared expert and the routed symmetric
W4A16 experts (``csrc/rocm/mono_ffn_rdna2.cu``). The all-reduce stays outside.

The kernel runs only inside a FULL CUDA graph (pure decode) with at most 24
rows; prefill, mixed and eager steps, and every unsupported layer, run
today's modules. The MTP drafter never takes it. The decision is made at run
time inside one opaque custom op, so torch.compile sees a single op and the
branch follows the actual batch, not the trace-time size.

Fail-stop: a timed-out grid barrier leaves a sticky record; later launches
exit at once and fill the FFN output with NaN instead of returning stale data.
"""

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MAX_ROWS = 24
_WORKSPACES: dict[int, torch.Tensor] = {}
_LOGGED: set = set()


def mono_ffn_requested(vllm_config) -> bool:
    extra = getattr(vllm_config, "additional_config", None) or {}
    return bool(isinstance(extra, dict) and extra.get("rdna_mono_ffn", False))


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
        ws = torch.zeros(nbytes + 4096, dtype=torch.uint8, device=device)
        _WORKSPACES[idx] = ws
    return ws


class MonoFFN:
    """Per-layer view of the weights the kernel reads; None if unsupported."""

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
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
            self.base, 1, 0, self.eps, 2000.0,
        )
        return res_out, out, inj_out


def _in_full_graph() -> bool:
    """True while a FULL (pure decode) CUDA graph is being captured.

    Model Runner V2 captures FULL graphs by running the model with
    cudagraph_runtime_mode NONE inside torch.cuda.graph; PIECEWISE capture runs
    with mode PIECEWISE. Eager steps are not capturing. FULL replays do not
    re-enter Python, so the capture-time choice is what replays."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not torch.cuda.is_current_stream_capturing():
        return False
    if not is_forward_context_available():
        return False
    return get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE


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
    use = (
        0 < m <= MAX_ROWS
        and hidden_states.dtype == torch.float16
        and mono.ready
        and _in_full_graph()
    )
    key = (m, use)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("rdna_mono_ffn: %d rows -> %s", m, "mono kernel" if use else "modules")
    if use:
        from vllm.distributed import tensor_model_parallel_all_reduce

        res_out, partial, inj_out = mono.run(hidden_states, attn_out, injection)
        return res_out, tensor_model_parallel_all_reduce(partial), inj_out
    hs, block_input, inj = layer.mlp_hyper_connection.combine_and_mix(
        hidden_states, attn_out, injection
    )
    return hs, layer.mlp(block_input), inj


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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 / K2 layout: the per-layer weight views the persistent kernel reads, the
per-device workspace, the rdna_ar attach for the in-kernel all-reduce and the
graph-mode helper shared by the parts."""

import torch

from vllm.logger import init_logger

from . import OPTS

logger = init_logger(__name__)

MAX_ROWS = 24  # widest batch K1 / K2 are built for
_WORKSPACES: dict[int, torch.Tensor] = {}
_AR_MAX: dict[int, int] = {}  # per device: rdna_ar slot bytes once attached
ZEROS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
LOGGED: set = set()


def mode() -> str:
    """The graph mode of the current call: full, piecewise or eager.

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


# graph modes K1 / K2 run in (``k2_modes``), filled by set_modes() at install
MODES: set[str] = set()


def set_modes() -> None:
    MODES.clear()
    MODES.update(x.strip() for x in str(OPTS.k2_modes).split(",") if x.strip())


def workspace(device: torch.device) -> torch.Tensor:
    """One zero-initialized workspace per device, allocated outside capture."""
    from . import debug

    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACES.get(idx)
    if ws is None:
        ops = torch.ops._rocm_C
        nbytes = int(ops.mono_k2_rdna2_workspace_bytes(2560, 320, 336, 512, 160, 640))
        # explicit coarse-grained allocation, not the caching allocator (whose
        # blocks can be fine-grained under HSA_FORCE_FINE_GRAIN_PCIE=1)
        ws = ops.mono_k2_rdna2_workspace_alloc(nbytes + 8192, idx)
        _WORKSPACES[idx] = ws
        probe = torch.empty(1, dtype=torch.uint8, device=device)
        logger.info(
            "mono k2_rdna2: workspace %d B on cuda:%d, HSA global flags %d "
            "(caching allocator: %d; 2 fine, 4 coarse, 8 extended fine)",
            nbytes + 8192,
            idx,
            ops.mono_k2_rdna2_mem_flags(ws),
            ops.mono_k2_rdna2_mem_flags(probe),
        )
        # stats / tuning block right after the kernel's workspace
        st = ws[nbytes : nbytes + debug.STAT_WORDS * 8].view(torch.int64)
        sc = debug.STAT_ROWS * debug.STAT_COLS
        if OPTS.k2_ar:
            _attach_ar(idx)
        st[sc + 0] = -1  # entry_min
        st[sc + 5] = OPTS.k2_sleep
        st[sc + 6] = OPTS.k2_flags
        if OPTS.timing:
            st[sc + 3] = 1  # enabled
            debug.STATS[idx] = st
    return ws


def _attach_ar(idx: int) -> None:
    """Point the K2 launches of this device at the TP group's rdna_ar one-shot
    buffers; the kernel then returns the all-reduced output."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        comm = getattr(get_tp_group().device_communicator, "rdna_ar_comm", None)
        handle = comm.handle if comm is not None and not comm.disabled else -1
        _AR_MAX[idx] = int(torch.ops._rocm_C.mono_k2_rdna2_set_ar(handle, idx))
    except Exception as e:  # noqa: BLE001
        _AR_MAX[idx] = 0
        logger.warning("mono k2_rdna2 ar: not attached (%s)", e)
    logger.info(
        "mono k2_rdna2 ar: cuda:%d %s",
        idx,
        f"fused all-reduce up to {_AR_MAX[idx]} B" if _AR_MAX[idx] else "off",
    )


def ar_fused(x: torch.Tensor) -> bool:
    """Mirrors the kernel's rule for doing the all-reduce itself."""
    if not OPTS.k2_ar:
        return False
    lim = _AR_MAX.get(x.device.index if x.device.index is not None else 0, 0)
    return lim > 0 and x.numel() * x.element_size() <= lim


class MonoLayer:
    """Per-decoder-layer view of the weights K1 / K2 read (resolved on the
    first eager call, after process_weights_after_loading). ``ready`` stays
    False, with a warning, for a geometry the kernel does not cover."""

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.first = getattr(layer, "layer_idx", None) == 0
        self.ready = False
        self.hc_ok = False
        self.reason = ""

    def setup(self) -> bool:
        if self.ready or self.reason:
            return self.ready
        try:
            self._setup()
            self.ready = True
        except Exception as e:  # unsupported geometry/quant: stay on modules
            self.reason = str(e)
            logger.warning_once(
                "mono k2_rdna2 disabled for %s: %s", type(self.layer).__name__, e
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
        if (
            getattr(down, "weight_i8", None) is not None
            or getattr(up, "weight_i8", None) is not None
        ):
            raise ValueError("int8 HC shadows are not supported")
        experts = getattr(mlp.experts, "routed_experts", mlp.experts)
        qm = getattr(experts, "quant_method", None) or getattr(
            mlp.experts, "_quant_method", None
        )
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
        if getattr(experts, "top_k", 10) != 10 or not getattr(
            experts, "renormalize", True
        ):
            raise ValueError("top-10 with renormalization required")
        self.t = dict(
            norm_w=hc.hc_norm.weight,
            wd=down.weight,
            wu=up.weight,
            wr=router,
            se_w1=gu,
            se_w2=dn,
            se_wg=wg.reshape(-1).contiguous(),
            w13=w13,
            s13=experts.w13_weight_scale,
            z13=experts.w13_qzeros,
            w2=experts.w2_weight_packed,
            s2=experts.w2_weight_scale,
            z2=experts.w2_qzeros,
        )
        self.base = base
        self.eps = float(hc.config.rms_norm_eps)
        self.ws = workspace(router.device)
        # K1: the attention-side HC module, same geometry as the MLP one
        ahc = layer.attn_hyper_connection
        adown = ahc.input_mix_weight_down_block_inject
        aup = ahc.input_mix_weight_up
        self.hc_ok = (
            ahc.hc_count == 4
            and ahc.use_combine
            and getattr(adown, "weight_i8", None) is None
            and getattr(aup, "weight_i8", None) is None
            and adown.weight.shape == down.weight.shape
            and aup.weight.shape == up.weight.shape
        )
        if self.hc_ok:
            self.th = dict(norm_w=ahc.hc_norm.weight, wd=adown.weight, wu=aup.weight)
            self.eps_a = float(ahc.config.rms_norm_eps)
            idx = router.device.index
            if (
                idx not in ZEROS
            ):  # (output, injection) for layers with no pending combine
                ZEROS[idx] = (
                    torch.zeros(
                        MAX_ROWS, 2560, dtype=torch.float16, device=router.device
                    ),
                    torch.zeros(MAX_ROWS, 4, dtype=torch.float16, device=router.device),
                )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp RDNA2 MonoKernel switch (``--monokernel``): CPU checks.

The flag and its env alias set ``kernel_config.monokernel``, off by default;
every deployment the kernels do not cover is refused at startup with
``--monokernel <reason>``; the switch adds no parameters and leaves the loaded
values alone.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import KernelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.models.qwen4_exp.amd import mono_decode_rdna2 as md
from vllm.models.qwen4_exp.amd.mono.rdna2 import OPTS, Options
from vllm.utils.argparse_utils import FlexibleArgumentParser

FLASH_NEXT = (
    "/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/"
    "snapshots/0939125b929543a783ce700c90e36dd1a575c00c"
)


def _config(
    monokernel=True, arch="Qwen4ExpForConditionalGeneration", v2=True, pp=1, knobs=None
):
    extra = {} if knobs is None else {"rocm_mono_decode": knobs}
    return SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        model_config=SimpleNamespace(architectures=[arch]),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        use_v2_model_runner=v2,
        additional_config=extra,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.fixture
def deployment(monkeypatch):
    """A deployment the kernels take: gfx10x, every op built."""
    monkeypatch.setattr(md, "_on_gfx10x", lambda: True)
    monkeypatch.setattr(md, "ops_refusal", lambda: None)
    saved = Options()
    saved.assign(OPTS)
    yield
    OPTS.assign(saved)


# ---- the switch -----------------------------------------------------------


def test_off_by_default():
    assert KernelConfig().monokernel is False
    assert md.options(_config(monokernel=False)) is None


def test_flag_parses_into_engine_args():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert parser.parse_args([]).monokernel is False
    assert parser.parse_args(["--monokernel"]).monokernel is True
    assert EngineArgs.monokernel is False  # the dataclass default


def test_flag_turns_on_every_part_with_the_validated_defaults():
    o = md.options(_config())
    assert (o.k1, o.k2, o.gdn, o.qsa, o.timing) == (True, True, True, True, False)
    assert (o.k1_max_rows, o.k2_max_rows, o.gdn_max_rows) == (24, 8, 64)


def test_knobs_dict_and_unknown_keys():
    o = md.options(_config(knobs={"k2_max_rows": 4, "gdn": False, "k1_max_rows": 99}))
    assert (o.k2_max_rows, o.gdn, o.k1_max_rows) == (4, False, 24)
    with pytest.raises(ValueError, match="unknown rocm_mono_decode keys"):
        md.options(_config(knobs={"rdna_mono_ffn": True}))
    # the knobs are only read with the switch on
    assert md.options(_config(monokernel=False, knobs={"bogus": 1})) is None


def _engine_config(monkeypatch, *args, env=None):
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or not os.path.isdir(FLASH_NEXT):
        pytest.skip("needs ROCm and the Flash-Next checkpoint")
    from vllm.platforms.rocm import on_gfx10x

    if not on_gfx10x():
        pytest.skip("needs a gfx10x GPU")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    if env is not None:
        monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", env)
    else:
        monkeypatch.delenv("VLLM_ROCM_MONO_DECODE", raising=False)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(
        [
            "--model",
            FLASH_NEXT,
            "--max-model-len",
            "4096",
            "--dtype",
            "float16",
            "--language-model-only",
            *args,
        ]
    )
    return EngineArgs.from_cli_args(ns).create_engine_config()


@pytest.mark.parametrize(
    "args, env, expected",
    [((), None, False), (("--monokernel",), None, True), ((), "1", True)],
    ids=["off", "flag", "env-alias"],
)
def test_flag_and_env_alias_reach_kernel_config(monkeypatch, args, env, expected):
    """End to end through create_engine_config (API-server side): the flag and
    VLLM_ROCM_MONO_DECODE=1 both set kernel_config.monokernel."""
    assert _engine_config(monkeypatch, *args, env=env).kernel_config.monokernel is (
        expected
    )


# ---- refusals -------------------------------------------------------------


@pytest.mark.parametrize(
    "patch, cfg, reason",
    [
        ({"_on_gfx10x": lambda: False}, {}, "needs a gfx10x"),
        ({}, {"arch": "Qwen3NextForCausalLM"}, "supports Qwen4Exp only"),
        ({}, {"v2": False}, "needs Model Runner V2"),
        ({}, {"pp": 2}, "pipeline parallelism"),
        ({"ops_refusal": lambda: "needs the _rocm_C ops ['x']"}, {}, "_rocm_C ops"),
    ],
    ids=["not-gfx10x", "not-qwen4exp", "not-mrv2", "pp", "ops"],
)
def test_unsupported_deployment_is_refused(monkeypatch, deployment, patch, cfg, reason):
    for name, value in patch.items():
        monkeypatch.setattr(md, name, value)
    with pytest.raises(ValueError, match=rf"^--monokernel .*{reason}.*\.$"):
        md.install(_config(**cfg))


def test_mtp_drafter_config_is_accepted(deployment):
    """The MTP drafter's own VllmConfig (arch Qwen4ExpMTP) carries the flag."""
    md.check_config(_config(arch="Qwen4ExpMTP"))
    assert md.config_refusal(_config(arch="Qwen4ExpMTP")) is None


def test_missing_ops_are_named(monkeypatch):
    monkeypatch.setattr(md, "REQUIRED_OPS", ("mono_does_not_exist_rdna2",))
    assert "mono_does_not_exist_rdna2" in md.ops_refusal()


def test_api_server_check_leaves_other_gpus_and_models_alone(monkeypatch):
    monkeypatch.setattr(md, "_on_gfx10x", lambda: False)
    md.check_config(_config(arch="DeepseekV41ForCausalLM"))  # another MonoKernel
    with pytest.raises(ValueError, match="^--monokernel needs a gfx10x"):
        md.check_config(_config())


class CompressedTensorsWNA16RDNA2MoEMethod:  # name-matched stand-in
    def __init__(self, symmetric=True):
        self.symmetric = symmetric


class _Experts(nn.Module):
    def __init__(self, qm):
        super().__init__()
        self.quant_method = qm
        self.w13_weight_packed = nn.Parameter(
            torch.randint(0, 99, (4, 8, 16), dtype=torch.int32), requires_grad=False
        )


class _Layer(nn.Module):
    """The parts of a Qwen4Exp decoder layer the switch looks at."""

    def __init__(self, qm):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _Experts(qm)
        self.mlp.gate = nn.Linear(16, 4, bias=False)
        self.attn_hyper_connection = nn.Linear(8, 8)
        self.mlp_hyper_connection = nn.Linear(8, 8)


@pytest.mark.parametrize(
    "qm, w4a8, reason",
    [
        (SimpleNamespace(), False, "got SimpleNamespace"),
        (CompressedTensorsWNA16RDNA2MoEMethod(symmetric=False), False, "asymmetric"),
        (CompressedTensorsWNA16RDNA2MoEMethod(), True, "got W4A8"),
    ],
    ids=["other-method", "asymmetric", "w4a8"],
)
def test_moe_not_symmetric_w4a16_rdna2_is_refused(
    monkeypatch, deployment, qm, w4a8, reason
):
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: w4a8)
    with pytest.raises(ValueError, match=rf"^--monokernel needs the MoE .*{reason}"):
        md.layer_hook(_Layer(qm), _config(), "model.layers.3", True)


# ---- parameters -----------------------------------------------------------


def _state(m: nn.Module):
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_switch_adds_no_parameters_and_keeps_values(monkeypatch, deployment):
    """named_parameters() and the loaded values are the same with the switch
    on or off: the per-layer MonoKernel state lives in the forward context,
    not in the module tree."""
    import vllm.model_executor.layers.fused_moe.experts.rdna2_w4a16_moe as moe

    monkeypatch.setattr(moe, "resolve_w4a8_moe", lambda: False)
    torch.manual_seed(0)
    off_layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    on_layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    on_layer.load_state_dict(off_layer.state_dict())

    off_cfg, on_cfg = _config(monokernel=False), _config()
    md.install(off_cfg)  # no-op without the switch
    assert md.layer_hook(off_layer, off_cfg, "model.layers.3", True) == (None, None)
    OPTS.assign(md.options(on_cfg))
    names = md.layer_hook(on_layer, on_cfg, "model.layers.3", True)
    assert names == ("model.layers.3.mono_rdna2",) * 2
    assert not off_cfg.compilation_config.static_forward_context
    assert list(on_cfg.compilation_config.static_forward_context) == [names[0]]

    assert [(n, p.shape) for n, p in off_layer.named_parameters()] == [
        (n, p.shape) for n, p in on_layer.named_parameters()
    ]
    on_state, off_state = _state(on_layer), _state(off_layer)
    assert on_state.keys() == off_state.keys()
    assert all(torch.equal(on_state[k], off_state[k]) for k in on_state)


def test_drafter_and_dense_layers_keep_their_modules(deployment):
    cfg = _config()
    OPTS.assign(md.options(cfg))
    layer = _Layer(CompressedTensorsWNA16RDNA2MoEMethod())
    assert md.layer_hook(layer, cfg, "mtp.layers.0", True) == (None, None)
    assert md.layer_hook(layer, cfg, "model.layers.0", False) == (None, None)


def test_gdn_and_qsa_parts_follow_the_switch(monkeypatch):
    assert md.qsa_part(_config(monokernel=False)) is None
    assert md.qsa_part(_config(knobs={"qsa": False})) is None
    from vllm.models.qwen4_exp.amd.mono.rdna2 import qsa

    assert md.qsa_part(_config()) is qsa


def test_hw_queue_budget(monkeypatch):
    monkeypatch.setattr(md, "_on_gfx10x", lambda: True)
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    md.set_worker_hw_queue_budget(_config(monokernel=False))
    assert os.environ["GPU_MAX_HW_QUEUES"] == "2"
    md.set_worker_hw_queue_budget(_config())
    assert os.environ["GPU_MAX_HW_QUEUES"] == "4"
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    md.set_worker_hw_queue_budget(_config(knobs={"hw_queues": 0}))
    assert os.environ["GPU_MAX_HW_QUEUES"] == "2"

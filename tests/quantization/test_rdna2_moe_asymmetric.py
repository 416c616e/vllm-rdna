# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native RDNA2 MoE path must decline asymmetric checkpoints."""

import pytest

pytest.importorskip("compressed_tensors")

from compressed_tensors.quantization import QuantizationArgs  # noqa: E402

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E402, E501
    rocm_moe_rdna,
)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_asymmetric_moe_is_not_native_rdna2(num_bits):
    args = QuantizationArgs(
        num_bits=num_bits, symmetric=False, strategy="group", group_size=32
    )
    assert rocm_moe_rdna.is_supported(args) is False

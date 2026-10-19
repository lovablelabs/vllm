# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 descales broadcast for the FlashAttention kernels keep a unit stride on
the head dim, which FA4 requires when it compiles a variant."""

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import _expand_descale


@pytest.mark.parametrize("shape", [(1, 1), (4, 1), (1, 2), (4, 2)])
def test_expand_descale_keeps_a_unit_head_stride(shape: tuple[int, int]) -> None:
    descale = _expand_descale(torch.tensor(0.5), shape)
    assert descale.shape == shape
    assert torch.all(descale == 0.5)
    # FA4 makes a descale contiguous only when its last stride is not 1, and a
    # tensor of size-1 dims counts as contiguous whatever its strides.
    normalized = descale if descale.stride(-1) == 1 else descale.contiguous()
    assert normalized.stride(-1) == 1


def test_expand_descale_broadcasts_per_head_scales() -> None:
    descale = _expand_descale(torch.tensor([0.5, 2.0]), (3, 2))
    assert descale.tolist() == [[0.5, 2.0]] * 3

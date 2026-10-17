# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StagedWriteTensor / FusedStagedWriter with list, ndarray and element writes."""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("uva", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mixed_chunks(uva: bool, dtype: torch.dtype):
    device = torch.device("cuda")
    t = StagedWriteTensor((4, 64), dtype, device, uva_instead_of_gpu=uva)
    expected = torch.zeros(4, 64, dtype=dtype)
    a = np.arange(40, dtype=np.int64) * 3
    t.stage_write(0, 0, a)
    expected[0, :40] = torch.from_numpy(a).to(dtype)
    src = [7, 8, 9]
    t.stage_write(2, 5, src)
    src.append(10)  # staging copied the list
    expected[2, 5:8] = torch.tensor([7, 8, 9], dtype=dtype)
    t.stage_write(3, 0, np.empty(0, dtype=np.int32))  # empty writes are dropped
    t.stage_write(1, 0, {11: 1, 12: 2}.keys())
    expected[1, :2] = torch.tensor([11, 12], dtype=dtype)
    t.stage_write_elem(3, 42)
    expected[3, 0] = 42
    t.apply_write()
    torch.cuda.synchronize()
    assert torch.equal(t.gpu.cpu(), expected)
    assert not t._staged_write_contents and t._staged_write_len == 0


def test_fused_writer_chunks():
    device = torch.device("cuda")
    tensors = [StagedWriteTensor((2, 32), torch.int32, device) for _ in range(2)]
    tensors[0].stage_write(1, 3, np.arange(10, dtype=np.int32))
    tensors[1].stage_write(0, 0, [5, 6])
    tensors[1].stage_write(1, 30, np.array([1, 2], dtype=np.int64))
    writer = FusedStagedWriter(device, max_writes=8)
    ptrs = torch.tensor(
        [t.gpu.data_ptr() for t in tensors], dtype=torch.int64, device=device
    )
    strides = torch.tensor(
        [t.gpu.stride(0) for t in tensors], dtype=torch.int64, device=device
    )
    writer.apply(tensors, ptrs, strides)
    torch.cuda.synchronize()
    e0 = torch.zeros(2, 32, dtype=torch.int32)
    e0[1, 3:13] = torch.arange(10, dtype=torch.int32)
    e1 = torch.zeros(2, 32, dtype=torch.int32)
    e1[0, :2] = torch.tensor([5, 6], dtype=torch.int32)
    e1[1, 30:32] = torch.tensor([1, 2], dtype=torch.int32)
    assert torch.equal(tensors[0].gpu.cpu(), e0)
    assert torch.equal(tensors[1].gpu.cpu(), e1)

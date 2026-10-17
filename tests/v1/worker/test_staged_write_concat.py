# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged-write chunk concatenation matches np.asarray for every chunk kind."""

import numpy as np
import pytest

from vllm.v1.worker.gpu.buffer_utils import _concat_chunks


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_concat_chunks_matches_asarray(dtype) -> None:
    rng = np.random.default_rng(0)
    chunks = [
        rng.integers(0, 150_000, 43_000).tolist(),  # a long prompt
        [1, 2, 3],  # short list: numpy's path
        np.arange(5, dtype=np.int64),
        [True, False] * 200,
        list(range(300)),
    ]
    expected = np.concatenate([np.asarray(c, dtype=dtype) for c in chunks])
    actual = _concat_chunks(chunks, dtype)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_concat_chunks_falls_back_to_numpy_rules() -> None:
    floats = [0.5 * i for i in range(300)]  # array("i") refuses floats
    np.testing.assert_array_equal(
        _concat_chunks([floats], np.int32), np.asarray(floats, dtype=np.int32)
    )
    with pytest.raises(OverflowError):
        _concat_chunks([[2**40] * 300], np.int32)

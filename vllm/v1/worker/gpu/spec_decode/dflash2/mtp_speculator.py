# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafting through the target's MTP layer (``DFlash2MTPDraftModel``).

Each step:

1. The MTP layer's draft prefill over the target's batch writes its KV cache
   for the verified rows, from the target's multi-stream hidden states and
   with the target's attention metadata (``MTPBlockSpeculator.propose``).
2. DFlash lays out the query block: the bonus token and ``K`` mask tokens at
   positions ``s .. s + K``.
3. The block pass runs the MTP layer over that block at ``s - 1 .. s - 1 + K``
   and keeps every row's mixer output for the dense stack's input bridge.
4. Heads with the position-1 pass run the MTP layer once more, on the first
   draft token at position ``s``.
5. The dense DFlash2 pass drafts from the bridged inputs; the first draft
   position (and the second, for ``pos1_source`` "mtp") takes the MTP layer's
   top-k, scored by its logits alone.

The block and position-1 passes write the MTP layer's K/V rows at speculative
positions, which the next step's prefill rewrites before any query reads them.
They replay FULL CUDA graphs over private slot-mapping buffers: the shared
buffer holds the dense pass's query slots.
"""

import copy
from typing import Any

import torch
import torch.nn as nn

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash2_mtp import (
    get_phase_e_config,
    get_pos1_source,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def mtp_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    """The runner's config with the target's MTP layer as a one-step drafter.

    ``num_speculative_tokens`` stays the head's ``K``: the MTP layer's
    attention then takes the ``1 + K`` block rows of a request as a decode
    batch, as the target's verification rows.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    config = copy.copy(vllm_config)
    config.speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=speculative_config.num_speculative_tokens,
        target_model_config=vllm_config.model_config,
        target_parallel_config=vllm_config.parallel_config,
    )
    return config


@triton.jit
def _merged_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    hidden_ptr,
    seq_lens_ptr,
    anchor_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    target_hidden_ptr,
    target_hidden_stride,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    mask_token_id,
    num_rows,
    max_num_reqs,
    hidden_size,
    BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    # Axis 1: one program per row that may be accepted copies its streams.
    row = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    q_start = tl.load(target_query_start_loc_ptr + req)
    q_end = tl.load(target_query_start_loc_ptr + req + 1)
    num_valid = q_end - q_start - tl.load(num_rejected_ptr + req)
    m_start = req * num_rows
    if row < num_valid:
        src = target_hidden_ptr + (q_start + row).to(tl.int64) * target_hidden_stride
        dst = hidden_ptr + (m_start + row).to(tl.int64) * hidden_size
        for h in range(0, hidden_size, HIDDEN_BLOCK):
            cols = h + tl.arange(0, HIDDEN_BLOCK)
            mask = cols < hidden_size
            tl.store(dst + cols, tl.load(src + cols, mask=mask), mask=mask)
    if row == 0:
        # As the draft prefill: the last verified row's input is the token
        # sampled after it.
        req_state = tl.load(idx_mapping_ptr + req)
        if tl.load(num_sampled_ptr + req) > 0:
            next_token = tl.load(last_sampled_ptr + req_state).to(tl.int32)
        else:
            next_token = tl.load(next_prefill_tokens_ptr + req_state).to(tl.int32)
        first_pos = tl.load(target_positions_ptr + q_start)
        for i in range(0, num_rows, BLOCK):
            j = i + tl.arange(0, BLOCK)
            in_rows = j < num_rows
            shifted = j < num_valid - 1
            token = tl.load(
                target_input_ids_ptr + q_start + j + 1, mask=in_rows & shifted, other=0
            )
            token = tl.where(
                shifted, token, tl.where(j == num_valid - 1, next_token, mask_token_id)
            )
            tl.store(input_ids_ptr + m_start + j, token, mask=in_rows)
            tl.store(positions_ptr + m_start + j, first_pos + j, mask=in_rows)
        tl.store(anchor_ptr + req, (m_start + num_valid - 1).to(tl.int64))
        tl.store(seq_lens_ptr + req, first_pos + num_rows)
        if req == num_reqs - 1:
            # Requests a CUDA graph pads the batch with: empty, anchor row 0.
            for i in range(num_reqs, max_num_reqs, BLOCK):
                k = i + tl.arange(0, BLOCK)
                tl.store(seq_lens_ptr + k, 0, mask=k < max_num_reqs)
                tl.store(anchor_ptr + k, 0, mask=k < max_num_reqs)


class _PrivateSlotBlockTables:
    """BlockTables that hands out a pass's private slot mappings at capture.

    A FULL graph keeps the slot-mapping buffer the metadata builders get at
    capture time, so a pass that fills its own buffer is captured on it.
    """

    def __init__(self, block_tables: BlockTables, slot_mappings: torch.Tensor):
        self._block_tables = block_tables
        self._slot_mappings = slot_mappings

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self._slot_mappings.fill_(PAD_SLOT_ID)
        return self._slot_mappings[:, :num_tokens]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._block_tables, name)


class MTPBlockSpeculator(MTPSpeculator):
    """The target's MTP layer, run as the first layer of a DFlash2 head."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        num_block_rows: int,
        use_pos1_pass: bool,
        varlen_prefill: bool,
    ):
        super().__init__(vllm_config, device)
        if self.draft_logits is not None:
            raise ValueError("DFlash2MTPDraftModel supports greedy drafting only")
        # One draft step: the prefill writes the KV cache of the verified
        # rows, and the block pass replaces the decode steps.
        self.num_speculative_steps = 1
        self.num_block_rows = num_block_rows
        self.use_pos1_pass = use_pos1_pass
        # Whether the target verifies under adaptive verification, whose
        # batches the prefill replays.
        self.varlen_prefill = varlen_prefill
        # self.hidden_size is the target's multi-stream width.
        self.mixed_hidden_size = self.draft_model_config.get_hidden_size()

        max_num_rows = self.max_num_reqs * num_block_rows
        self.block_hidden_states = torch.zeros(
            max_num_rows, self.hidden_size, dtype=self.dtype, device=device
        )
        # Mixer output of every block row: the input of the dense stack's bridge.
        self.block_mixed = torch.zeros(
            max_num_rows, self.mixed_hidden_size, dtype=self.dtype, device=device
        )
        self.anchor_rows = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * num_block_rows
        )
        self.query_start_loc_pos1 = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        if use_pos1_pass:
            self.anchor_multi_hidden = torch.zeros(
                self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
            )
            self.pos1_mixed = torch.zeros(
                self.max_num_reqs,
                self.mixed_hidden_size,
                dtype=self.dtype,
                device=device,
            )

        # Decode-only steps run the prefill and the block pass as one forward:
        # [verified rows | K mask rows | padding], 1 + 2K rows per request at
        # the positions after the request's first verified one.
        self.merged_rows = 2 * num_block_rows - 1
        max_num_merged_rows = self.max_num_reqs * self.merged_rows
        self.merged_input_ids = torch.zeros(
            max_num_merged_rows, dtype=torch.int32, device=device
        )
        self.merged_hidden_states = torch.zeros(
            max_num_merged_rows, self.hidden_size, dtype=self.dtype, device=device
        )
        self.merged_anchors = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.query_start_loc_merged = self.query_start_loc_pos1 * self.merged_rows
        self.block_offsets = torch.arange(num_block_rows, device=device)
        self.use_merged_pass = False

        self.block_slot_mappings: torch.Tensor | None = None
        self.pos1_slot_mappings: torch.Tensor | None = None
        self.merged_slot_mappings: torch.Tensor | None = None
        self.block_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.pos1_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.merged_cudagraph_manager: SpeculatorCudaGraphManager | None = None

    def bind_head(
        self, head: nn.Module, head_input_buffers: InputBuffers, top_k: int
    ) -> None:
        """Wire the DFlash2 head: its bridge parameters, its query block and
        the number of candidates per draft position."""
        predictor = self.model.model
        if predictor.hc_count * predictor.hidden_size != self.hidden_size:
            raise ValueError(
                "The MTP layer's multi-stream width does not match the target's"
            )
        if getattr(predictor, "num_mtp_layers", 1) != 1:
            raise ValueError("DFlash2MTPDraftModel drafts through one MTP layer")
        self.head = head
        self.head_input_buffers = head_input_buffers
        self.mask_token_id = head.model.mask_token_id
        self.top_k = top_k
        if self.use_pos1_pass:
            self.root_ids = torch.zeros(
                self.max_num_reqs, top_k, dtype=torch.int64, device=self.device
            )
            self.root_logits = torch.zeros(
                self.max_num_reqs, top_k, dtype=torch.float32, device=self.device
            )

    def load_overrides(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load a stage-2 head's fine-tuned tensors over the MTP layer's."""
        loaded = self.model.load_weights(iter(weights))
        if not loaded:
            raise ValueError("None of the head's MTP tensors matched the MTP layer")
        logger.info(
            "Loaded %d fine-tuned MTP tensors from the DFlash2 head into %d "
            "parameters of the MTP layer.",
            len(weights),
            len(loaded),
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        if block_tables.cp_size > 1:
            raise NotImplementedError(
                "DFlash2MTPDraftModel does not support context parallelism"
            )
        self.block_slot_mappings = torch.full_like(
            block_tables.slot_mappings, PAD_SLOT_ID
        )
        if self.use_pos1_pass:
            self.pos1_slot_mappings = torch.full_like(
                block_tables.slot_mappings, PAD_SLOT_ID
            )
        self.use_merged_pass = self._ring_holds_merged_rows(kv_cache_config)
        if self.use_merged_pass:
            self.merged_slot_mappings = torch.full_like(
                block_tables.slot_mappings, PAD_SLOT_ID
            )
            # Builders of its own whose decode threshold takes the merged
            # rows, which the other passes' builders would treat as prefills.
            self.merged_attn_groups, _, _ = init_attn_backend(
                kv_cache_config,
                self.attn_vllm_config,
                self.device,
                active_layer_names=self.draft_attn_layer_names,
            )
            for groups in self.merged_attn_groups:
                for group in groups:
                    for builder in group.metadata_builders:
                        if builder.reorder_batch_threshold is not None:
                            builder.reorder_batch_threshold = max(
                                builder.reorder_batch_threshold, self.merged_rows
                            )

    def _ring_holds_merged_rows(self, kv_cache_config: KVCacheConfig) -> bool:
        """Whether every QSA indexer ring holds the merged pass's rows plus
        the open group's committed keys."""
        ratio = int(
            getattr(
                self.vllm_config.model_config.hf_text_config,
                "indexer_compress_ratio",
                1,
            )
            or 1
        )
        capacities = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = (
                spec.kv_cache_specs.values()
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else [spec]
            )
            capacities += [
                x.block_size for x in specs if isinstance(x, CircularBufferSpec)
            ]
        need = self.merged_rows + ratio - 1
        if capacities and min(capacities) < need:
            logger.info(
                "The merged MTP pass is off: the QSA ring holds %d rows, it needs %d.",
                min(capacities),
                need,
            )
            return False
        return True

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The prefill replays the target's batch shapes.
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_block_rows,
            varlen_decode=self.varlen_prefill,
        )
        self.decode_cudagraph_manager = None
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        self.block_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, cudagraph_mode, self.num_block_rows
        )
        if self.use_pos1_pass:
            self.pos1_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device, cudagraph_mode, decode_query_len=1
            )
        if self.use_merged_pass:
            self.merged_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device, cudagraph_mode, self.merged_rows
            )

    def capture(self) -> None:
        # The prefill's graphs; num_speculative_steps == 1 has no decode graphs.
        super().capture()
        assert self.block_cudagraph_manager is not None
        assert self.block_slot_mappings is not None
        self.block_cudagraph_manager.capture(
            self._block_pass,
            self.model_state,
            self.input_buffers,
            _PrivateSlotBlockTables(self.block_tables, self.block_slot_mappings),
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing MTP block CUDA graphs",
        )
        if self.use_pos1_pass:
            assert self.pos1_cudagraph_manager is not None
            assert self.pos1_slot_mappings is not None
            self.pos1_cudagraph_manager.capture(
                self._pos1_pass,
                self.model_state,
                self.input_buffers,
                _PrivateSlotBlockTables(self.block_tables, self.pos1_slot_mappings),
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing MTP position-1 CUDA graphs",
            )
        if self.use_merged_pass:
            assert self.merged_cudagraph_manager is not None
            assert self.merged_slot_mappings is not None
            self.merged_cudagraph_manager.capture(
                self._merged_pass,
                self.model_state,
                self.input_buffers,
                _PrivateSlotBlockTables(self.block_tables, self.merged_slot_mappings),
                self.merged_attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing MTP merged-pass CUDA graphs",
            )

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        # Only the KV cache is kept: the block pass drafts the first token.
        self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )

    @torch.inference_mode()
    def _block_pass(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            input_ids = self.head_input_buffers.input_ids[:num_tokens]
            sample_hidden_states, multi_hidden_states = self.model(
                input_ids=None,
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.block_hidden_states[:num_tokens],
                inputs_embeds=self.head.mtp_input_embeds(
                    self.model.embed_input_ids, input_ids
                ),
                mask_rows=input_ids == self.mask_token_id,
                mask_hidden_states=self.head.mtp_head.no_hidden,
            )
        self.block_mixed[:num_tokens].copy_(sample_hidden_states)
        if self.use_pos1_pass:
            num_anchors = num_tokens // self.num_block_rows
            self.anchor_multi_hidden[:num_anchors].copy_(
                multi_hidden_states[self.anchor_rows[:num_anchors]]
            )

    @torch.inference_mode()
    def _pos1_pass(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        num_speculative_steps: int | None = None,
    ) -> None:
        anchor_rows = self.anchor_rows[:num_tokens]
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            root_ids, root_logits = self.model.logits_processor.get_top_k_tokens(
                self.model.lm_head, self.block_mixed[anchor_rows], self.top_k
            )
            self.root_ids[:num_tokens].copy_(root_ids)
            self.root_logits[:num_tokens].copy_(root_logits)
            sample_hidden_states, _ = self.model(
                input_ids=root_ids[:, 0].contiguous(),
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.anchor_multi_hidden[:num_tokens],
            )
        self.pos1_mixed[:num_tokens].copy_(sample_hidden_states)
        # Row 2 of the block is the dense row that drafts position 1.
        self.block_mixed.index_copy_(0, anchor_rows + 2, sample_hidden_states)

    @torch.inference_mode()
    def _merged_pass(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            input_ids = self.merged_input_ids[:num_tokens]
            sample_hidden_states, multi_hidden_states = self.model(
                input_ids=None,
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.merged_hidden_states[:num_tokens],
                inputs_embeds=self.head.mtp_input_embeds(
                    self.model.embed_input_ids, input_ids
                ),
                mask_rows=input_ids == self.mask_token_id,
                mask_hidden_states=self.head.mtp_head.no_hidden,
            )
        # The anchor and its K mask rows are the block pass's rows.
        num_reqs = num_tokens // self.merged_rows
        anchors = self.merged_anchors[:num_reqs]
        window = (anchors[:, None] + self.block_offsets).flatten()
        self.block_mixed[: num_reqs * self.num_block_rows].copy_(
            sample_hidden_states[window]
        )
        if self.use_pos1_pass:
            self.anchor_multi_hidden[:num_reqs].copy_(multi_hidden_states[anchors])

    def root_candidates(self, num_reqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k ids and logits of the first draft position (the anchor row)."""
        if self.use_pos1_pass:
            # Computed by the position-1 pass, whose input is the top-1.
            return self.root_ids[:num_reqs], self.root_logits[:num_reqs]
        return self.model.logits_processor.get_top_k_tokens(
            self.model.lm_head,
            self.block_mixed[self.anchor_rows[:num_reqs]],
            self.top_k,
        )

    def run_dummy_passes(self, num_reqs: int) -> None:
        """The block and position-1 forwards without attention (profiling)."""
        num_tokens = num_reqs * self.num_block_rows
        self._prepare_eplb_forward(num_tokens)
        self._block_pass(num_reqs, num_tokens, None, None, None)
        if self.use_pos1_pass:
            self._prepare_eplb_forward(num_reqs)
            self._pos1_pass(num_reqs, num_reqs, None, None, None)
        if self.use_merged_pass:
            num_tokens = num_reqs * self.merged_rows
            self._prepare_eplb_forward(num_tokens)
            self._merged_pass(num_reqs, num_tokens, None, None, None)

    def _dispatch(
        self,
        cudagraph_manager: SpeculatorCudaGraphManager | None,
        num_reqs: int,
        num_rows_per_req: int,
        dp_sync_state: DPSyncState | None,
        is_profile: bool,
    ) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
        batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync_state, num_reqs, num_rows_per_req)
            if dp_sync_state is not None
            else (None, num_reqs * num_rows_per_req)
        )
        batch_desc, batch_sync = dispatch_cg_and_sync_dp(
            cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=num_rows_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync_state=batch_sync,
        )
        num_tokens_across_dp = (
            batch_sync.num_tokens_across_dp if batch_sync is not None else None
        )
        return batch_desc, num_tokens_across_dp

    def _run_pass(
        self,
        forward_fn,
        cudagraph_manager: SpeculatorCudaGraphManager | None,
        private_slot_mappings: torch.Tensor,
        input_batch: InputBatch,
        num_rows_per_req: int,
        dp_sync_state: DPSyncState | None,
        is_profile: bool,
        attn_groups: list[list[AttentionGroup]] | None = None,
    ) -> None:
        """Run one pass over ``num_rows_per_req`` rows per request whose
        positions, seq_lens and query_start_loc are in ``self.input_buffers``."""
        num_reqs = input_batch.num_reqs
        batch_desc, num_tokens_across_dp = self._dispatch(
            cudagraph_manager, num_reqs, num_rows_per_req, dp_sync_state, is_profile
        )
        num_tokens_padded = batch_desc.num_tokens
        self.block_tables.compute_slot_mappings(
            self.idx_mapping[:num_reqs],
            self.input_buffers.query_start_loc[: num_reqs + 1],
            self.input_buffers.positions,
            num_tokens_padded,
            out=private_slot_mappings,
        )
        num_attn_tokens = (
            num_tokens_padded
            if batch_desc.cg_mode == CUDAGraphMode.FULL
            else num_reqs * num_rows_per_req
        )
        # Any row of the pass sees at most K positions past the target's.
        attn_metadata = self._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=self.arange_np[: num_reqs + 1] * num_rows_per_req,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=self.num_block_rows - 1,
            slot_mappings=private_slot_mappings[:, :num_attn_tokens],
            attn_groups=attn_groups,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            private_slot_mappings[:, :num_tokens_padded], self.kv_cache_config
        )
        self._prepare_eplb_forward(num_reqs * num_rows_per_req)
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert cudagraph_manager is not None
            cudagraph_manager.run_fullgraph(batch_desc)
        else:
            forward_fn(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings_by_layer,
                num_tokens_across_dp,
                batch_desc.cg_mode,
            )

    @torch.inference_mode()
    def propose_block(
        self,
        input_batch: InputBatch,
        target_multi_hidden_states: torch.Tensor,
        dp_sync_state: DPSyncState | None,
        is_profile: bool,
    ) -> None:
        """The MTP layer over the head's query block, after the prefill (whose
        last_token_indices are this step's anchors)."""
        assert self.block_slot_mappings is not None
        num_reqs = input_batch.num_reqs
        num_tokens = num_reqs * self.num_block_rows
        head_buffers = self.head_input_buffers
        # The head's rows sit at s + k; the MTP layer's at s - 1 + k.
        torch.sub(
            head_buffers.positions[:num_tokens],
            1,
            out=self.input_buffers.positions[:num_tokens],
        ).clamp_(min=0)
        self.input_buffers.query_start_loc.copy_(head_buffers.query_start_loc)
        torch.sub(head_buffers.seq_lens, 1, out=self.input_buffers.seq_lens).clamp_(
            min=0
        )
        self.block_hidden_states.view(self.max_num_reqs, self.num_block_rows, -1)[
            :num_reqs, 0
        ] = target_multi_hidden_states[self.last_token_indices[:num_reqs]]
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_block_rows, self.max_model_len
        )
        self._run_pass(
            self._block_pass,
            self.block_cudagraph_manager,
            self.block_slot_mappings,
            input_batch,
            self.num_block_rows,
            dp_sync_state,
            is_profile,
        )

    @torch.inference_mode()
    def propose_merged(
        self,
        input_batch: InputBatch,
        target_multi_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        is_profile: bool,
    ) -> None:
        """The draft prefill and the block pass as one forward, for a
        decode-only batch on one DP rank. Every request's 1 + 2K rows fit the
        slots the scheduler allocated: its K + 1 verification rows and the
        head's K + 1 lookahead slots."""
        assert self.merged_slot_mappings is not None
        num_reqs = input_batch.num_reqs
        self._copy_request_inputs(
            num_reqs, input_batch.idx_mapping, temperature, seeds, dummy_run=False
        )
        _merged_inputs_kernel[(num_reqs, self.num_block_rows)](
            self.merged_input_ids,
            self.input_buffers.positions,
            self.merged_hidden_states,
            self.input_buffers.seq_lens,
            self.merged_anchors,
            input_batch.input_ids,
            input_batch.positions,
            target_multi_hidden_states,
            target_multi_hidden_states.stride(0),
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            self.mask_token_id,
            self.merged_rows,
            self.max_num_reqs,
            self.hidden_size,
            BLOCK=32,
            HIDDEN_BLOCK=2048,
        )
        self.input_buffers.query_start_loc.copy_(self.query_start_loc_merged)
        self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(
            num_reqs * self.merged_rows
        )
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_block_rows, self.max_model_len
        )
        self._run_pass(
            self._merged_pass,
            self.merged_cudagraph_manager,
            self.merged_slot_mappings,
            input_batch,
            self.merged_rows,
            None,
            is_profile,
            attn_groups=self.merged_attn_groups,
        )

    @torch.inference_mode()
    def propose_pos1(
        self,
        input_batch: InputBatch,
        dp_sync_state: DPSyncState | None,
        is_profile: bool,
    ) -> None:
        """The MTP layer on the first draft token at position s (the head's
        anchor position), one row per request, after the block pass."""
        assert self.pos1_slot_mappings is not None
        num_reqs = input_batch.num_reqs
        positions = self.input_buffers.positions[:num_reqs]
        # The anchor's position s; it sees the context, the block's anchor
        # row at s - 1 and itself.
        positions.copy_(self.head_input_buffers.positions[self.anchor_rows[:num_reqs]])
        self.input_buffers.seq_lens[:num_reqs].copy_(positions + 1)
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.query_start_loc_pos1[: num_reqs + 1]
        )
        self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(num_reqs)
        self._run_pass(
            self._pos1_pass,
            self.pos1_cudagraph_manager,
            self.pos1_slot_mappings,
            input_batch,
            1,
            dp_sync_state,
            is_profile,
        )


class DFlash2MTPSpeculator(DFlash2Speculator):
    _speculator_name = "DFlash2-MTP"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.draft_logits is not None:
            raise ValueError("DFlash2MTPDraftModel supports greedy drafting only")
        phase_e_config = get_phase_e_config(self.draft_model_config.hf_config)
        self.pos1_source = get_pos1_source(phase_e_config)
        if self.pos1_source is not None and self.num_speculative_steps < 2:
            raise ValueError(
                "The position-1 pass of DFlash2MTPDraftModel needs "
                "num_speculative_tokens >= 2"
            )
        self.mtp = MTPBlockSpeculator(
            mtp_vllm_config(self.vllm_config),
            device,
            num_block_rows=self.num_query_per_req,
            use_pos1_pass=self.pos1_source is not None,
            varlen_prefill=self.enable_adaptive_verification,
        )
        # Whether this step's MTP layer ran the merged pass.
        self._merged_step = False

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._dense_attn_layer_names = self.draft_attn_layer_names
        # Loaded after the dense head: its draft layers are the MTP layer's.
        self.mtp.load_model(target_model)
        self.mtp.bind_head(self.model, self.input_buffers, self.top_k)
        if self.model.mtp_overrides:
            self.mtp.load_overrides(self.model.mtp_overrides)
            self.model.mtp_overrides = []
        # Both heads score with the target's lm_head, so the MTP layer's top-k
        # can come from the dense candidate GEMM.
        mtp_logits, dense_logits = (
            self.mtp.model.logits_processor,
            self.model.candidate_logits_processor,
        )
        if (
            self.mtp.model.lm_head is not self.model.lm_head
            or mtp_logits.scale != dense_logits.scale
            or mtp_logits.soft_cap != dense_logits.soft_cap
        ):
            raise ValueError(
                "DFlash2MTPDraftModel needs the target's lm_head and logit "
                "scaling for both the dense stack and the MTP layer"
            )
        # The runner allocates and validates every draft layer's KV cache.
        self.draft_attn_layer_names = (
            self._dense_attn_layer_names | self.mtp.draft_attn_layer_names
        )

    def set_eplb_state(self, eplb_state: EplbState) -> None:
        super().set_eplb_state(eplb_state)
        self.mtp.set_eplb_state(eplb_state)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        # The dense stack's attention groups cover its own layers only.
        draft_attn_layer_names = self.draft_attn_layer_names
        self.draft_attn_layer_names = self._dense_attn_layer_names
        try:
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
        finally:
            self.draft_attn_layer_names = draft_attn_layer_names
        self.mtp.set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        self.mtp.init_cudagraph_manager(cudagraph_mode)

    def capture(self) -> None:
        self.mtp.capture()
        super().capture()

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hc_count * hidden_size]: the target's multi-stream states
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        if not aux_hidden_states:
            raise RuntimeError("DFlash2MTPDraftModel needs the target's aux states")
        self._merged_step = (
            self.mtp.use_merged_pass
            and not dummy_run
            and not input_batch.has_prefill
            and self.dp_size == 1
        )
        if self._merged_step:
            self.mtp.propose_merged(
                input_batch,
                last_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                is_profile,
            )
        else:
            self._propose_mtp_prefill(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )

    def _propose_mtp_prefill(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None,
        dummy_run: bool,
        skip_attn_for_dummy_run: bool,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
        is_profile: bool,
    ) -> None:
        # The MTP prefill runs on the target's attention metadata and slot
        # mappings, before the head lays out its query block.
        self.mtp.propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            None,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )

    def run_query_passes(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        dp_sync_state: DPSyncState | None,
        dummy_run: bool,
        skip_attn: bool,
        is_profile: bool,
    ) -> None:
        if skip_attn:
            self.mtp.run_dummy_passes(input_batch.num_reqs)
            return
        if not self._merged_step:
            self.mtp.propose_block(
                input_batch, last_hidden_states, dp_sync_state, is_profile
            )
        if self.pos1_source is not None:
            self.mtp.propose_pos1(input_batch, dp_sync_state, is_profile)

    def _inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        return self.model.bridged_input_embeds(
            self.input_buffers.input_ids[:num_tokens],
            self.mtp.block_mixed[:num_tokens],
        )

    def _compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = hidden_states.shape[0]
        if self.pos1_source == "mtp":
            # The extra MTP row drafts position 1; score it in the same GEMM.
            hidden_states = hidden_states.clone()
            hidden_states[:, 1] = self.mtp.pos1_mixed[:num_reqs]
        candidate_ids, unary_logits = super()._compute_candidates(hidden_states)
        root_ids, root_logits = self.mtp.root_candidates(num_reqs)
        candidate_ids[:, 0] = root_ids
        unary_logits[:, 0] = root_logits
        return candidate_ids, unary_logits

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        scores = super()._score_candidates(candidate_ids, unary_logits, hidden_states)
        # The positions the MTP layer drafts keep its greedy choice; the walk
        # takes candidate 0 at step 0, the predecessor slot step 1 reads.
        scores[:, 0, 0] = unary_logits[:, 0]
        if self.pos1_source == "mtp":
            scores[:, 1, 0] = unary_logits[:, 1]
        return scores

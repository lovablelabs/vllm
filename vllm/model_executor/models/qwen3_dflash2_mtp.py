# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash2 head that drafts through the target's MTP layer.

Each step the target's MTP layer runs over the DFlash query block: the anchor
row (the last verified position ``s - 1``, with the target's pre-final-mixer
streams and the bonus token) and the ``K`` mask rows (a learned ``no_hidden``
in place of the streams, the mask embedding in place of the token), at
positions ``s - 1 .. s - 1 + K``. The anchor row's MTP prediction is the first
draft token, and every row's mixer output is bridged into the dense DFlash2
stack's input::

    h_in = base + bridge(RMSNorm(mixed)) * rms(base)

where ``base`` is the DFlash2 input embedding of the row. Heads trained with
``phaseE.pos1_mtp_pass`` run the MTP layer once more on the first draft token
at position ``s``; that row predicts the second draft position and replaces
the mixer output bridged into its dense row.

The MTP layer is the target checkpoint's. Stage-2 heads (``phaseE.stage`` 2,
``stage2_mode`` "nonexpert") ship fine-tuned copies of its non-expert tensors
as ``mtp_head.mtp.*`` / ``mtp_head.tail.*``, mapped to checkpoint names by
``phaseE.mtp_trained_tensors``. The speculator owns the MTP layer
(``DFlash2MTPSpeculator``) and loads those tensors onto it.
"""

import re
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from .utils import maybe_prefix

logger = init_logger(__name__)

_BRIDGE_PREFIX = "mtp_head."
_MTP_OVERRIDE_PREFIXES = ("mtp_head.mtp.", "mtp_head.tail.")
# The trainer keeps some tensors concatenated along dim 0 and names their
# checkpoint sources "cat([a, b], 0)" (optionally followed by a shape note).
_CONCAT_REF = re.compile(r"^cat\(\[(.+?)\],\s*0\)")


def get_phase_e_config(hf_config) -> dict:
    """The head's ``phaseE`` config section ({} for a stage-1 export)."""
    return getattr(hf_config, "phaseE", None) or {}


def get_pos1_source(phase_e_config: dict) -> str | None:
    """Where the second draft position comes from: "mtp" (the extra MTP row),
    "dense" (the dense stack), or None without the position-1 pass."""
    if not phase_e_config.get("pos1_mtp_pass"):
        return None
    source = (phase_e_config.get("pos1") or {}).get("pos1_source", "mtp")
    if source not in ("mtp", "dense"):
        raise ValueError(f"Unknown phaseE.pos1.pos1_source {source!r}")
    return source


def split_mtp_overrides(
    overrides: dict[str, torch.Tensor], table: dict[str, dict]
) -> list[tuple[str, torch.Tensor]]:
    """(checkpoint name, tensor) pairs of a stage-2 head's MTP tensors.

    ``table`` maps each exported name to its ``checkpoint_tensor``, ``shape``
    and ``dtype``; a ``cat([a, b], 0)`` source is split into equal chunks along
    dim 0 in the listed order.
    """
    missing = sorted(table.keys() - overrides.keys())
    unknown = sorted(overrides.keys() - table.keys())
    if missing or unknown:
        raise ValueError(
            "phaseE.mtp_trained_tensors does not match the exported tensors: "
            f"missing {missing}, unknown {unknown}"
        )
    pairs: list[tuple[str, torch.Tensor]] = []
    for name in sorted(overrides):
        entry = table[name]
        tensor = overrides[name]
        if (
            list(tensor.shape) != list(entry["shape"])
            or str(tensor.dtype).replace("torch.", "") != entry["dtype"]
        ):
            raise ValueError(
                f"{name}: {list(tensor.shape)} {tensor.dtype} does not match "
                f"phaseE.mtp_trained_tensors ({entry['shape']} {entry['dtype']})"
            )
        ref = entry["checkpoint_tensor"].strip()
        match = _CONCAT_REF.match(ref)
        sources = [ref] if match is None else [s.strip() for s in match[1].split(",")]
        if tensor.shape[0] % len(sources):
            raise ValueError(f"{name}: dim 0 is not divisible into {sources}")
        pairs.extend(zip(sources, tensor.chunk(len(sources), dim=0)))
    return pairs


class DFlash2MTPBridge(nn.Module):
    """``no_hidden`` and the bridge from MTP mixer outputs to the dense input."""

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        rms_norm_eps: float,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.no_hidden = nn.Parameter(
            torch.zeros(hc_count, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.bridge_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.bridge = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "bridge"),
            return_bias=False,
        )

    def forward(self, base: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        delta = self.bridge(self.bridge_norm(mixed)).float()
        scale = base.float().pow(2).mean(-1, keepdim=True).sqrt()
        return (base.float() + delta * scale).to(base.dtype)


class DFlash2MTPQwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    """The dense DFlash2 stack plus the bridge into it from the MTP layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_config = vllm_config.model_config.hf_text_config
        hc_count = int(getattr(target_config, "hc_count", 1) or 1)
        if hc_count <= 1:
            raise ValueError(
                "DFlash2MTPDraftModel needs a hyper-connection target (hc_count > 1)"
            )
        if target_config.hidden_size != self.config.hidden_size:
            raise ValueError(
                "DFlash2MTPDraftModel needs the target's hidden size "
                f"({target_config.hidden_size}), got {self.config.hidden_size}"
            )
        self.phase_e_config = get_phase_e_config(self.config)
        stage = int(self.phase_e_config.get("stage", 1))
        if stage == 2 and self.phase_e_config.get("stage2_mode") != "nonexpert":
            raise ValueError(
                f"phaseE.stage2_mode {self.phase_e_config.get('stage2_mode')!r} is "
                "not supported; stage-2 heads must ship non-expert MTP tensors only"
            )
        if stage not in (1, 2):
            raise ValueError(f"Unknown phaseE.stage {stage}")
        for feature in ("anchor_bridge_layers", "dense_reveal_rows"):
            if self.phase_e_config.get(feature):
                raise ValueError(f"phaseE.{feature} is not supported")
        self.pos1_source = get_pos1_source(self.phase_e_config)
        self.mtp_head = DFlash2MTPBridge(
            hidden_size=self.config.hidden_size,
            hc_count=hc_count,
            rms_norm_eps=self.config.rms_norm_eps,
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "mtp_head"),
        )
        # (checkpoint name, tensor) pairs for the speculator's MTP layer.
        self.mtp_overrides: list[tuple[str, torch.Tensor]] = []

    def bridged_input_embeds(
        self, input_ids: torch.Tensor, mixed: torch.Tensor
    ) -> torch.Tensor:
        """The dense stack's input: the DFlash2 embedding plus the bridged MTP
        mixer output ``mixed`` [T, H] of the same rows."""
        return self.mtp_head(self.model.embed_input_ids(input_ids), mixed)

    def mtp_input_embeds(
        self, embed_tokens: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """The MTP layer's input embeddings: the target's token embedding, the
        (unscaled) mask embedding on the mask rows."""
        embeds = embed_tokens(input_ids)
        is_mask = (input_ids == self.model.mask_token_id).unsqueeze(-1)
        return torch.where(is_mask, self.model.mask_embedding.to(embeds.dtype), embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        bridge_weights: list[tuple[str, torch.Tensor]] = []
        overrides: dict[str, torch.Tensor] = {}

        def dense_weights():
            for name, loaded_weight in weights:
                if name.startswith(_MTP_OVERRIDE_PREFIXES):
                    overrides[name] = loaded_weight.detach().cpu().clone()
                elif name.startswith(_BRIDGE_PREFIX):
                    bridge_weights.append((name, loaded_weight))
                else:
                    yield name, loaded_weight

        super().load_weights(dense_weights())
        if not self.model.has_separate_mask_embedding:
            raise ValueError(
                "DFlash2MTPDraftModel needs mask_embedding.pt: the MTP layer's "
                "mask rows share the trained mask embedding"
            )

        params = dict(self.mtp_head.named_parameters())
        loaded = set()
        for name, loaded_weight in bridge_weights:
            param_name = name.removeprefix(_BRIDGE_PREFIX)
            param = params.get(param_name)
            if param is None or param.shape != loaded_weight.shape:
                raise ValueError(
                    f"Unexpected Phase E tensor {name} {tuple(loaded_weight.shape)}"
                )
            getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
            loaded.add(param_name)
        if loaded != params.keys():
            raise ValueError(
                f"Missing Phase E tensors: {sorted(params.keys() - loaded)}"
            )

        table = self.phase_e_config.get("mtp_trained_tensors") or {}
        if overrides or table:
            if int(self.phase_e_config.get("stage", 1)) != 2:
                raise ValueError("MTP tensors in a head whose phaseE.stage is not 2")
            self.mtp_overrides = split_mtp_overrides(overrides, table)


EntryClass = DFlash2MTPQwen3ForCausalLM

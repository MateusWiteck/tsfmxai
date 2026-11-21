"""Chronos-2 with an absolute channel-index embedding added at the input.

Chronos-2 group attention has no positional information, so its outputs are
equivariant to channel permutations and cannot name a series by its index.
This variant adds a learned embedding of each series' position inside its
group to every token of that series (context, register and future tokens)
before the first encoder layer. The residual stream carries it into every
group attention layer. Rows of one group must be contiguous and ordered by
channel, which is the layout produced by ``adapt_grouped_batch``.

The flag is stored as ``config.channel_index_embedding`` (the number of
channel positions), so ``from_pretrained`` rebuilds the same architecture.
Without the flag the model is the unmodified Chronos-2.
"""
from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn

from chronos.chronos2.model import Chronos2Model


def channel_index_from_groups(group_ids: torch.Tensor) -> torch.Tensor:
    """Position of every row inside its group, assuming contiguous groups."""
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    if group_ids.numel() == 0:
        return group_ids.clone()
    starts = torch.ones_like(group_ids, dtype=torch.bool)
    starts[1:] = group_ids[1:] != group_ids[:-1]
    if torch.unique(group_ids).numel() != int(starts.sum()):
        raise ValueError("Rows of each group must be contiguous")
    rows = torch.arange(group_ids.numel(), device=group_ids.device)
    first_row = torch.cummax(torch.where(starts, rows, torch.zeros_like(rows)), dim=0).values
    return rows - first_row


class ChannelIndexedEncoder(nn.Module):
    """Add the channel-index embedding to the encoder input, then run Chronos-2."""

    def __init__(self, encoder: nn.Module, channels: int, d_model: int):
        super().__init__()
        self.inner = encoder
        self.channels = channels
        self.channel_embedding = nn.Embedding(channels, d_model)

    def forward(self, inputs_embeds, *, group_ids, **kwargs):
        index = channel_index_from_groups(group_ids)
        if int(index.max()) >= self.channels:
            raise ValueError(f"A group has more than {self.channels} series")
        inputs_embeds = inputs_embeds + self.channel_embedding(index)[:, None, :].to(inputs_embeds.dtype)
        return self.inner(inputs_embeds, group_ids=group_ids, **kwargs)


class IndexedChronos2Model(Chronos2Model):
    """Chronos-2 whose encoder input carries the absolute channel index."""

    def __init__(self, config):
        super().__init__(config)
        channels = getattr(config, "channel_index_embedding", None)
        if channels:
            if type(channels) is not int or channels < 1:
                raise ValueError("channel_index_embedding must be a positive integer")
            self.encoder = ChannelIndexedEncoder(self.encoder, channels, config.d_model)
            # Same initialization rule the model applies to its own embeddings.
            self._init_weights(self.encoder.channel_embedding)


def load_backbone(folder: str | Path) -> Chronos2Model:
    """Load an unmodified or channel-indexed Chronos-2 checkpoint."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    cls = IndexedChronos2Model if config.get("channel_index_embedding") else Chronos2Model
    return cls.from_pretrained(str(folder))

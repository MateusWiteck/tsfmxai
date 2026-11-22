"""Frozen skeleton teacher targets for AST channels of a synthetic batch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoder
from tsfmxai.symbolic_regression.canonical_release import codec_contract


class EquationTargetTeacher:
    def __init__(self, source: str, revision: str, *, channels=6, max_lag=16,
                 max_degree=4, device="cpu", on_overflow="raise"):
        if not source:
            raise ValueError("Pin the skeleton-only teacher source")
        if on_overflow not in ("mask", "raise"):
            raise ValueError("on_overflow must be 'mask' or 'raise'")
        self.on_overflow = on_overflow
        local = Path(source)
        if not local.exists() and (len(revision) != 40 or
                                   any(c not in "0123456789abcdef" for c in revision.lower())):
            raise ValueError("Pin an immutable 40-character Hugging Face commit")
        self.encoder = CanonicalEncoder.from_pretrained(
            (source, "transformer_skeleton"), revision=revision or None).to(device).eval()
        self.codec = CanonicalCodec(channels, 1, max_lag, max_degree)
        if self.encoder.source_embedding.num_embeddings != self.codec.vocab_size:
            raise ValueError("Teacher vocabulary differs from generator vocabulary")
        # Equal sizes can still map ids to different atoms (6 series x 16 lags and
        # 3 series x 32 lags both give 197 tokens), so the whole codec must match.
        released = getattr(self.encoder, "release_config", {}).get("codec")
        expected = codec_contract(self.codec)
        if released is None:
            raise ValueError("Teacher release has no codec block; cannot check token meaning")
        differing = sorted(k for k in expected if released.get(k) != expected[k])
        if differing:
            raise ValueError(f"Teacher codec differs from generator codec in {differing}")
        self.device = device
        self.dimension = self.encoder.dimension
        self.provenance = {"source": source, "revision": revision,
                           "variant": "transformer_skeleton", "on_overflow": on_overflow}
        if local.exists():
            folder = next((p for p in (local / "transformer_skeleton",
                                        local / "models" / "transformer_skeleton", local)
                           if (p / "encoder.safetensors").exists()), None)
            if folder is None:
                raise FileNotFoundError("Local teacher encoder.safetensors missing")
            self.provenance["encoder_sha256"] = hashlib.sha256(
                (folder / "encoder.safetensors").read_bytes()).hexdigest()

    @torch.no_grad()
    def targets(self, batch, *, scale_floor=1e-5, coverage=None):
        """Return [example,channel] masks, embeddings and context-scaled sigma.

        ``coverage``, if given, is a dict updated in place with, per treatment,
        how many AST channels received a skeleton target and how many were masked.
        """
        context = np.asarray(batch["context"], dtype=np.float32)
        examples, _, channels = context.shape
        if len(batch["metadata"]) != examples:
            raise ValueError("Metadata must contain one system per example")
        mask = np.zeros((examples, channels), dtype=bool)
        noise_mask = np.zeros((examples, channels), dtype=bool)
        raw_std = np.zeros((examples, channels), dtype=np.float32)
        rows, locations = [], []
        for example, metadata in enumerate(batch["metadata"]):
            system = metadata["system"]
            families = metadata.get("channel_families", ("ast",) * channels)
            for channel, family in enumerate(families):
                if family != "ast":
                    continue
                noise_mask[example, channel] = True
                raw_std[example, channel] = system.innovations[channel].std
                encoded = self.codec.encode(system.equations[channel])
                method = system.treatments[channel].method
                counts = (coverage.setdefault(method, {"covered": 0, "masked": 0})
                          if coverage is not None else None)
                if len(encoded["prefix_ids"]) > self.encoder.max_nodes:
                    # The frozen teacher has no learned positions beyond max_nodes.
                    if self.on_overflow == "raise":
                        raise ValueError(
                            f"A {method} equation has {len(encoded['prefix_ids'])} nodes; "
                            f"the teacher accepts {self.encoder.max_nodes}")
                    # Keep forecasting/noise supervision but mask this embedding.
                    if counts is not None:
                        counts["masked"] += 1
                    continue
                if counts is not None:
                    counts["covered"] += 1
                rows.append(encoded)
                locations.append((example, channel))
                mask[example, channel] = True
        embeddings = np.zeros((examples, channels, self.dimension), dtype=np.float32)
        if rows:
            width = max(len(row["prefix_ids"]) for row in rows)
            ids = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
            values = torch.zeros((len(rows), width), dtype=torch.float32, device=self.device)
            lengths = torch.empty(len(rows), dtype=torch.long, device=self.device)
            for index, row in enumerate(rows):
                length = len(row["prefix_ids"])
                ids[index, :length] = torch.tensor(row["prefix_ids"], device=self.device)
                values[index, :length] = torch.tensor(row["prefix_values"], device=self.device)
                lengths[index] = length
            _, _, root = self.encoder({"prefix_ids": ids, "prefix_values": values,
                                       "lengths": lengths})
            for (example, channel), vector in zip(locations, root.cpu().numpy()):
                embeddings[example, channel] = vector
        # Pinned Chronos-2 InstanceNorm uses population std of observed context
        # and substitutes eps=1e-5 only when scale is exactly zero.
        scale = context.std(axis=1)
        scale = np.where(scale == 0, scale_floor, scale)
        scaled_std = np.where(noise_mask, raw_std / scale, 0).astype(np.float32)
        return {"teacher_embedding": embeddings, "equation_target_mask": mask,
                "noise_std": scaled_std, "noise_mask": noise_mask}

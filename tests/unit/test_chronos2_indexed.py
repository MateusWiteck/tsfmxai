"""Channel-indexed Chronos-2 with one-step patches."""
import numpy as np
import pytest
import torch

from tsfmxai.models.chronos2_indexed import (
    ChannelIndexedEncoder, IndexedChronos2Model, channel_index_from_groups, load_backbone,
)
from tsfmxai.pipelines.chronos2_lmc import _inputs, scratch_model


def batch(seed=0, examples=2, channels=6):
    generator = torch.Generator().manual_seed(seed)
    return {"context": torch.randn(examples, 96, channels, generator=generator),
            "future": torch.randn(examples, 96, channels, generator=generator)}


def predictions(model, data):
    output = model(**_inputs(data, model, "cpu"))
    examples, _, channels = data["context"].shape
    return output.quantile_preds.reshape(examples, channels, *output.quantile_preds.shape[1:])


def test_channel_index_follows_contiguous_groups():
    groups = torch.tensor([0, 0, 0, 1, 1, 1, 7, 7])
    assert channel_index_from_groups(groups).tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    with pytest.raises(ValueError):
        channel_index_from_groups(torch.tensor([0, 1, 0]))


def test_one_step_patches_give_one_token_per_step():
    model = scratch_model(seed=1, tiny=True, patch_size=1, channel_index_embedding=6)
    assert model.chronos_config.input_patch_size == model.chronos_config.output_patch_size == 1
    captured = []
    model.encoder.register_forward_hook(lambda module, args, output: captured.append(output[0]))
    output = model(**_inputs(batch(), model, "cpu"))
    # 96 context steps + register token + 96 future steps per series.
    assert captured[0].shape[1] == 96 + 1 + 96
    assert output.quantile_preds.shape[-1] == 96
    assert torch.isfinite(output.loss)


def test_channel_index_breaks_permutation_equivariance_only_when_enabled():
    data = batch()
    order = torch.tensor([3, 1, 0, 2, 5, 4])
    shuffled = {key: value[:, :, order] for key, value in data.items()}
    for embedding, changes in ((6, True), (None, False)):
        model = scratch_model(seed=1, tiny=True, patch_size=1, channel_index_embedding=embedding)
        model.eval()
        with torch.no_grad():
            original = predictions(model, data)[:, order]
            permuted = predictions(model, shuffled)
        assert (not torch.allclose(original, permuted, atol=1e-6)) == changes


def test_save_and_load_keep_architecture_and_predictions(tmp_path):
    model = scratch_model(seed=1, tiny=True, patch_size=1, channel_index_embedding=6).eval()
    model.save_pretrained(tmp_path)
    loaded = load_backbone(tmp_path).eval()
    assert isinstance(loaded, IndexedChronos2Model)
    assert isinstance(loaded.encoder, ChannelIndexedEncoder)
    with torch.no_grad():
        np.testing.assert_allclose(predictions(model, batch()).numpy(),
                                   predictions(loaded, batch()).numpy(), atol=1e-6)


def test_default_backbone_is_the_unmodified_architecture():
    model = scratch_model(seed=1, tiny=True)
    assert model.chronos_config.input_patch_size == 16
    assert not isinstance(model.encoder, ChannelIndexedEncoder)


def test_more_series_than_index_positions_is_rejected():
    model = scratch_model(seed=1, tiny=True, patch_size=1, channel_index_embedding=6)
    with pytest.raises(ValueError, match="more than 6"):
        model(**_inputs(batch(channels=7), model, "cpu"))

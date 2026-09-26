"""Tests for the SigLIP 2 encoder: input spec, image parity and text handling."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from mcap_lancedb.embed import SiglipEncoder, _input_spec, _pooled, resolve_device


def test_input_spec_reads_size_and_filter() -> None:
    """A fixed-resolution processor gives its size and resampling filter."""
    processor = SimpleNamespace(
        size={"height": 384, "width": 384}, resample=Image.Resampling.BILINEAR
    )
    spec = _input_spec(processor)
    assert (spec.height, spec.width, spec.resample) == (
        384,
        384,
        Image.Resampling.BILINEAR,
    )


def test_input_spec_rejects_naflex() -> None:
    """A NaFlex processor has no size, and fails with a clear message."""
    processor = SimpleNamespace(size=None, resample=Image.Resampling.BILINEAR)
    with pytest.raises(ValueError, match="NaFlex"):
        _input_spec(processor)


def test_resolve_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto picks the best available backend; an explicit device always wins."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
    assert resolve_device("cuda").type == "cuda"


def test_encode_images_matches_the_hugging_face_processor(
    siglip2_encoder: SiglipEncoder,
) -> None:
    """Resizing on CPU and normalizing on device reproduces the processor path."""
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8))
    gradient = Image.fromarray(
        np.broadcast_to(np.linspace(0, 255, 1600, dtype=np.uint8), (900, 1600))
    ).convert("RGB")
    images = [noise, gradient]

    ours = siglip2_encoder.encode_images(images)
    processor = AutoProcessor.from_pretrained(siglip2_encoder.model_id)
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        output = siglip2_encoder.model.get_image_features(
            pixel_values=pixels.to(siglip2_encoder.device, siglip2_encoder.dtype)
        )
    reference = torch.nn.functional.normalize(_pooled(output).float(), dim=-1)
    cosine = (ours * reference.cpu().numpy()).sum(axis=1)
    assert cosine.min() > 0.99999


def test_encode_text_lowercases(siglip2_encoder: SiglipEncoder) -> None:
    """SigLIP 2 was trained on lowercase text; its Gemma tokenizer is cased."""
    upper, lower = siglip2_encoder.encode_text(["A Bus At A Stop", "a bus at a stop"])
    np.testing.assert_array_equal(upper, lower)


def test_encode_text_pads_to_a_fixed_length(siglip2_encoder: SiglipEncoder) -> None:
    """A query embeds the same alone or next to a longer one.

    SigLIP pools the last position without an attention mask, so padding to the
    longest query in the batch would change the short query's embedding.
    """
    alone = siglip2_encoder.encode_text(["a bus"])[0]
    batched = siglip2_encoder.encode_text(["a bus", "a long query " * 8])[0]
    assert alone @ batched > 0.99999


def test_embeddings_are_unit_length(siglip2_encoder: SiglipEncoder) -> None:
    """Text embeddings are L2-normalized and ``dim`` wide."""
    vectors = siglip2_encoder.encode_text(["a bus", "a truck"])
    assert vectors.shape == (2, siglip2_encoder.dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

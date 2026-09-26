"""SigLIP 2 image and text encoder, shared by the ingest pipeline and the viewer.

Images and text land in the same L2-normalized space, so a dot product is a
cosine similarity. Only fixed-resolution checkpoints are supported: they squash
every image to one square size, which lets the ingest pipeline resize on CPU and
hand the GPU actors model-ready pixels.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from mcap_lancedb import DEFAULT_MODEL

# SigLIP was trained on text padded to exactly 64 tokens. Shorter padding does
# not fail; it silently degrades retrieval.
TEXT_MAX_LENGTH = 64


@dataclass(frozen=True)
class ImageInputSpec:
    """How a checkpoint wants its images: squashed to a fixed size, no crop."""

    height: int
    width: int
    resample: Image.Resampling

    @property
    def nbytes(self) -> int:
        """Size of one image in bytes as packed ``uint8`` RGB."""
        return self.height * self.width * 3

    def prepare(self, image: Image.Image) -> np.ndarray:
        """Resize an image exactly as the checkpoint's processor would.

        Args:
            image: Any PIL image.

        Returns:
            A ``uint8`` array of shape ``(height, width, 3)``.
        """
        rgb = image.convert("RGB")
        return np.asarray(rgb.resize((self.width, self.height), self.resample))


def _input_spec(image_processor: Any) -> ImageInputSpec:  # noqa: ANN401
    """Read the input size and resampling filter from an image processor."""
    size = image_processor.size
    height, width = size["height"], size["width"]
    if not height or not width:
        msg = (
            f"Expected a fixed-resolution checkpoint, got image size {size}. "
            "NaFlex checkpoints are not supported."
        )
        raise ValueError(msg)
    # SigLIP 2 squashes with bilinear, SigLIP 1 with bicubic, so this is read
    # from the processor rather than assumed.
    return ImageInputSpec(
        int(height), int(width), Image.Resampling(image_processor.resample)
    )


def image_input_spec(model_id: str) -> ImageInputSpec:
    """Look up a checkpoint's image input spec without loading its weights.

    Args:
        model_id: Hugging Face model id.

    Returns:
        The size and resampling filter its processor uses.
    """
    # AutoImageProcessor needs torchvision in transformers 5; AutoProcessor
    # falls back to the PIL image processor, which is what the model card uses.
    return _input_spec(AutoProcessor.from_pretrained(model_id).image_processor)


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a torch device, preferring CUDA, then MPS, then CPU.

    Args:
        requested: An explicit device such as ``"cuda"`` or ``"cpu"``. ``None``
            or ``"auto"`` picks the best available one.

    Returns:
        The device to run on.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pooled(output: Any) -> torch.Tensor:  # noqa: ANN401
    """Extract pooled features from a ``get_*_features`` result.

    transformers 5 returns ``BaseModelOutputWithPooling``; earlier releases
    returned the tensor itself.
    """
    return output if isinstance(output, torch.Tensor) else output.pooler_output


class SiglipEncoder:
    """Embed images and text with a SigLIP 2 checkpoint.

    Attributes:
        model_id: Hugging Face model id.
        device: Device the model runs on.
        dtype: fp16 on CUDA, fp32 elsewhere.
        input_spec: Image size and resampling the checkpoint expects.
        dim: Embedding dimension, probed from a real forward pass.
    """

    def __init__(
        self, model_id: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        """Load the processor and model.

        Args:
            model_id: Hugging Face model id of a fixed-resolution checkpoint.
            device: Explicit device, or ``None`` to pick the best available.
        """
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(model_id)
        self._tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        self.input_spec = _input_spec(image_processor)

        # Pixel normalization runs on the device, so the CPU stage only resizes.
        def channel_tensor(values: Sequence[float]) -> torch.Tensor:
            return torch.tensor(values, device=self.device, dtype=self.dtype).view(
                1, 3, 1, 1
            )

        self._rescale = float(image_processor.rescale_factor)
        self._mean = channel_tensor(image_processor.image_mean)
        self._std = channel_tensor(image_processor.image_std)

        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        # Config field names differ between SigLIP variants; one forward pass
        # gives the true output width.
        self.dim = int(self.encode_text(["a photo"]).shape[1])

    @torch.inference_mode()
    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Embed images that are already at the model's input size.

        Args:
            pixels: ``uint8`` RGB of shape ``(n, height, width, 3)``, as produced
                by ``input_spec.prepare``.

        Returns:
            L2-normalized ``float32`` embeddings of shape ``(n, dim)``.
        """
        batch = torch.from_numpy(pixels).to(self.device).permute(0, 3, 1, 2)
        batch = (batch.to(self.dtype) * self._rescale - self._mean) / self._std
        return self._normalize(self.model.get_image_features(pixel_values=batch))

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed PIL images of any size.

        Args:
            images: Images to embed.

        Returns:
            L2-normalized ``float32`` embeddings of shape ``(n, dim)``.
        """
        return self.encode_pixels(
            np.stack([self.input_spec.prepare(i) for i in images])
        )

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text queries.

        Args:
            texts: Queries to embed. They are lowercased first, matching how
                SigLIP 2 was trained.

        Returns:
            L2-normalized ``float32`` embeddings of shape ``(n, dim)``.
        """
        # transformers' Siglip2Tokenizer lowercases on its own, but these
        # checkpoints load a plain GemmaTokenizer, which is case-sensitive.
        tokens = self._tokenizer(
            [text.lower() for text in texts],
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        return self._normalize(self.model.get_text_features(input_ids=input_ids))

    @staticmethod
    def _normalize(output: Any) -> np.ndarray:  # noqa: ANN401
        """Pool, L2-normalize in fp32, and move to host memory."""
        features = torch.nn.functional.normalize(_pooled(output).float(), dim=-1)
        return features.cpu().numpy()

"""Loading pretrained vision models and extracting their native pooled features.

Each model exposes a uniform interface: a ``preprocess`` step turning a PIL image
into a tensor, and an ``embed`` step turning a batch of such tensors into the
model's native pooled representation (DINOv2 CLS pooler, SigLIP attention-pooled
head, MAE mean-pooled patch tokens). Two backends are supported: HuggingFace
``transformers`` and ``timm``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from PIL.Image import Image
from torch import Tensor


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a pretrained model to load.

    Attributes:
        name: Short identifier used in outputs and cache filenames.
        backend: Either ``"transformers"`` or ``"timm"``.
        hf_id: Model id passed to the backend loader.
        pooling: Native pooling: ``"pooler"`` (transformers pooler_output),
            ``"head"`` (timm default head pooling), or ``"avg"`` (timm mean pool).
    """

    name: str
    backend: str
    hf_id: str
    pooling: str


class LoadedModel:
    """A loaded model with a uniform preprocess/embed interface."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        """Load ``spec`` onto ``device`` and prepare its preprocessing pipeline.

        On CUDA the model runs in bfloat16 so matmuls use the GPU's tensor cores
        (the forward pass is compute-bound, dominated by the largest model); the
        bf16 outputs are upcast to float32 before caching. CPU stays float32.
        """
        self.spec = spec
        self.device = device
        self._dtype = torch.bfloat16 if "cuda" in device else torch.float32
        if spec.backend == "transformers":
            self._init_transformers()
        elif spec.backend == "timm":
            self._init_timm()
        else:
            raise ValueError(f"Unknown backend {spec.backend!r}")

    def _init_transformers(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self.spec.hf_id)
        self._model = AutoModel.from_pretrained(self.spec.hf_id).eval().to(self.device, self._dtype)
        self._dim = self._model.config.hidden_size

    def _init_timm(self) -> None:
        import timm
        from timm.data import create_transform, resolve_data_config

        # Keep the checkpoint's native pooling/norm layout; "avg" pooling is done
        # manually over patch tokens (overriding global_pool breaks state-dict load).
        self._model = (
            timm.create_model(self.spec.hf_id, pretrained=True, num_classes=0)
            .eval()
            .to(self.device, self._dtype)
        )
        cfg = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**cfg)
        self._dim = self._model.num_features

    @property
    def dim(self) -> int:
        """Dimensionality of the model's pooled representation."""
        return self._dim

    def preprocess(self, image: Image) -> Float[Tensor, "c h w"]:
        """Convert a PIL image into the model's expected input tensor."""
        rgb = image.convert("RGB")
        if self.spec.backend == "transformers":
            return self._processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        return self._transform(rgb)

    @torch.no_grad()
    def embed(self, batch: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d"]:
        """Return native pooled representations for a preprocessed batch.

        Outputs are in the model's compute dtype (bf16 on CUDA); the caller
        upcasts to float32 before caching.
        """
        batch = batch.to(self.device, self._dtype)
        if self.spec.backend == "transformers":
            return self._model(pixel_values=batch).pooler_output
        if self.spec.pooling == "avg":
            tokens = self._model.forward_features(batch)
            return tokens[:, self._model.num_prefix_tokens :].mean(dim=1)
        return self._model(batch)

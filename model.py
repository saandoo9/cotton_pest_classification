# model.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from layercam import LayerCAM, crop_from_cam_bbox_adjusted


@dataclass(frozen=True)
class ClassInfo:
    """Optional metadata for a multiclass label."""
    name: str
    kind: str


RawImage = Union[np.ndarray, Image.Image]          # OpenCV ndarray (H,W,C) or PIL Image
PreprocessFn = Callable[[Image.Image], torch.Tensor]  # user preprocessing for multiclass crop


class DualStageCropClassifier(nn.Module):
    """
    Binary relevance filtering, Layer-CAM square cropping and classification.

    Accepts model instances or checkpoint paths. State dictionaries require a
    matching supplied architecture. Full-model pickle loading is opt-in.

    Preprocessing stays outside the model:
      - fit() expects preprocessed tensors from the user.
      - predict() expects x_bin as a preprocessed tensor and optionally raw_image + preprocess_multi.
    """


    # Default Layer-CAM target layer path for the binary model (string path resolver)
    BINARY_TARGET_LAYER_PATH = "features.-1"


    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        localization_threshold: float,
        binary_checkpoint: Optional[str] = None,
        multiclass_checkpoint: Optional[str] = None,
        binary_model: Optional[nn.Module] = None,
        multiclass_model: Optional[nn.Module] = None,
        target_layer_path: str = BINARY_TARGET_LAYER_PATH,
        relevant_class: int = 1,
        trust_checkpoint: bool = False,
        multiclass_input_size: Tuple[int, int] = (224, 224),
        class_index: Optional[Sequence[Dict[str, str]]] = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.localization_threshold = float(localization_threshold)
        if not 0 <= self.localization_threshold <= 1:
            raise ValueError("localization_threshold must lie in [0,1].")
        if relevant_class not in (0, 1):
            raise ValueError("relevant_class must be 0 or 1 for the two-output binary filter.")
        self.relevant_class = relevant_class
        self.multiclass_input_size = tuple(multiclass_input_size)

        self.binary_model = self._load_model(binary_checkpoint, binary_model, trust_checkpoint)
        self.multiclass_model = self._load_model(multiclass_checkpoint, multiclass_model, trust_checkpoint)
        target_layer = self._resolve_layer(self.binary_model, target_layer_path)
        self.layercam = LayerCAM(self.binary_model, target_layer)

        self.class_index: Optional[List[ClassInfo]] = None
        if class_index is not None:
            self.set_class_index(class_index)

    # ----------------------------
    # Utilities
    # ----------------------------

    def _load_model(self, path, model=None, trust_checkpoint=False) -> nn.Module:
        """Load strict state dictionaries, or explicitly trusted full models."""
        if path is not None:
            checkpoint = torch.load(path, map_location=self.device,
                                    weights_only=not trust_checkpoint)
            if isinstance(checkpoint, nn.Module):
                if model is not None:
                    raise ValueError("Supply either a full-model checkpoint or a model architecture, not both.")
                model = checkpoint
            else:
                if model is None:
                    raise ValueError("State-dict checkpoints need a matching model architecture.")
                state = checkpoint
                for key in ("model_state_dict", "state_dict"):
                    if isinstance(checkpoint, dict) and key in checkpoint:
                        state = checkpoint[key]
                        break
                model.load_state_dict(state, strict=True)
        if not isinstance(model, nn.Module):
            raise ValueError("Supply a model instance or a trusted full-model checkpoint.")
        return model.to(self.device).eval()

    @staticmethod
    def _resolve_layer(model: nn.Module, path: str) -> nn.Module:
        """
        Resolve a submodule given a dotted path.
        Supports integer indexing for containers (including negative indices).
        Example: "features.-1" -> model.features[-1]
        """
        cur: Any = model
        for token in path.split("."):
            if token == "":
                continue
            try:
                idx = int(token)
                cur = cur[idx]
                continue
            except ValueError:
                pass
            if not hasattr(cur, token):
                raise AttributeError(f'Cannot resolve "{path}": missing attribute "{token}".')
            cur = getattr(cur, token)
        if not isinstance(cur, nn.Module):
            raise TypeError(f'Resolved "{path}" but it is not an nn.Module.')
        return cur

    def set_class_index(self, class_index: Sequence[Dict[str, str]]) -> None:
        """Set multiclass label metadata."""
        parsed: List[ClassInfo] = []
        for item in class_index:
            name = item.get("nombre", item.get("name", ""))
            kind = item.get("tipo", item.get("kind", ""))
            parsed.append(ClassInfo(name=name, kind=kind))
        self.class_index = parsed

    @staticmethod
    def _to_pil(raw_image: RawImage) -> Image.Image:
        """Convert OpenCV ndarray or PIL image to RGB PIL image."""
        if isinstance(raw_image, Image.Image):
            img = raw_image
        elif isinstance(raw_image, np.ndarray):
            if raw_image.ndim == 2:
                img = Image.fromarray(raw_image).convert("RGB")
            elif raw_image.ndim == 3 and raw_image.shape[2] in (3, 4):
                if raw_image.shape[2] == 3:
                    rgb = raw_image[..., ::-1]  # BGR -> RGB
                    img = Image.fromarray(rgb)
                else:
                    rgba = raw_image[..., [2, 1, 0, 3]]  # BGRA -> RGBA
                    img = Image.fromarray(rgba).convert("RGB")
            else:
                raise ValueError("Unsupported ndarray shape for image.")
        else:
            raise TypeError("raw_image must be a numpy.ndarray (OpenCV) or PIL.Image.Image.")
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ----------------------------
    # Differentiable forward
    # ----------------------------

    def forward(self, x_bin: torch.Tensor, x_multi: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Differentiable forward for training (no LayerCAM cropping here).
        """
        out: Dict[str, torch.Tensor] = {"binary_logits": self.binary_model(x_bin)}
        if x_multi is not None:
            out["multiclass_logits"] = self.multiclass_model(x_multi)
        return out

    # ----------------------------
    # Training
    # ----------------------------

    def fit(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        *,
        epochs: int,
        criterion_binary: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        criterion_multiclass: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        train_binary: bool = True,
        train_multiclass: bool = True,
        lr_binary: float = 1e-4,
        lr_multiclass: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        grad_clip_norm: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Train binary and/or multiclass with different learning rates using param groups.

        Batch dict requirements:
          - If train_binary=True:  "x_bin", "y_bin"
          - If train_multiclass=True: "x_multi", "y_multi"
        """
        device = self.device
        self.to(device)
        self.train()

        if train_multiclass and criterion_multiclass is None:
            raise ValueError("criterion_multiclass must be provided when train_multiclass=True.")

        # One optimizer, two param groups -> different LRs
        param_groups: List[Dict[str, Any]] = []
        if train_binary:
            param_groups.append({"params": self.binary_model.parameters(), "lr": float(lr_binary)})
        if train_multiclass:
            param_groups.append({"params": self.multiclass_model.parameters(), "lr": float(lr_multiclass)})

        optimizer = optimizer_cls(param_groups, weight_decay=float(weight_decay))

        history: Dict[str, List[float]] = {"loss_binary": [], "loss_multiclass": [], "loss_total": []}

        for _ in range(int(epochs)):
            sum_bin = 0.0
            sum_multi = 0.0
            sum_total = 0.0
            steps = 0

            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss_total = 0.0

                if train_binary:
                    if "x_bin" not in batch or "y_bin" not in batch:
                        raise KeyError('Batch must include "x_bin" and "y_bin" when train_binary=True.')
                    x_bin = batch["x_bin"].to(device, non_blocking=True)
                    y_bin = batch["y_bin"].to(device, non_blocking=True)
                    logits_bin = self.binary_model(x_bin)
                    loss_bin = criterion_binary(logits_bin, y_bin)
                    loss_total = loss_total + loss_bin
                    sum_bin += float(loss_bin.item())

                if train_multiclass:
                    if "x_multi" not in batch or "y_multi" not in batch:
                        raise KeyError('Batch must include "x_multi" and "y_multi" when train_multiclass=True.')
                    x_multi = batch["x_multi"].to(device, non_blocking=True)
                    y_multi = batch["y_multi"].to(device, non_blocking=True)
                    logits_multi = self.multiclass_model(x_multi)
                    loss_multi = criterion_multiclass(logits_multi, y_multi)  # type: ignore[arg-type]
                    loss_total = loss_total + loss_multi
                    sum_multi += float(loss_multi.item())

                loss_total.backward()

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip_norm))

                optimizer.step()

                sum_total += float(loss_total.item())
                steps += 1

            denom = max(steps, 1)
            history["loss_binary"].append(sum_bin / denom)
            history["loss_multiclass"].append(sum_multi / denom)
            history["loss_total"].append(sum_total / denom)

        return history

    # ----------------------------
    # Inference
    # ----------------------------

    @torch.no_grad()
    def predict(
        self,
        x_bin: torch.Tensor,
        *,
        raw_image: Optional[RawImage] = None,
        preprocess_multi: Optional[PreprocessFn] = None,
        binary_no_object_class: Optional[int] = None,
        return_cam: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict using the pipeline.

        Required:
          - x_bin: (B,C,H,W) preprocessed tensor for the binary model.

        Optional for full pipeline (single image only):
          - raw_image: OpenCV ndarray or PIL image for ROI cropping.
          - preprocess_multi: function that preprocesses the cropped PIL image into a tensor
                             for the multiclass model.

        Returns:
          - Always returns binary predictions (batch supported).
          - Multiclass path runs only if batch size == 1 and required args are provided.
        """
        self.eval()
        device = self.device
        self.to(device)

        if x_bin.ndim != 4:
            raise ValueError("x_bin must be a 4D tensor of shape (B,C,H,W).")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Call predict outside torch.inference_mode(); Layer-CAM requires autograd.")

        if binary_no_object_class is None:
            binary_no_object_class = 1 - self.relevant_class
        if binary_no_object_class != 1 - self.relevant_class:
            raise ValueError("binary_no_object_class must complement relevant_class.")
        x_bin = x_bin.to(device)

        logits_bin = self.binary_model(x_bin)
        if logits_bin.ndim != 2 or logits_bin.shape[1] != 2:
            raise ValueError("The binary relevance filter must return two logits per image.")
        probs_bin = torch.softmax(logits_bin, dim=1)
        pred_bin = torch.argmax(logits_bin, dim=1)  # (B,)
        conf_bin = probs_bin.gather(1, pred_bin.view(-1, 1)).squeeze(1)  # (B,)

        result: Dict[str, Any] = {
            "binary_prediction": pred_bin.detach().cpu().tolist(),
            "binary_confidence": conf_bin.detach().cpu().tolist(),
            "has_object": [int(p.item()) != int(binary_no_object_class) for p in pred_bin],
        }

        # Only single-image mode supports LayerCAM crop here
        if x_bin.shape[0] != 1:
            return result

        has_object = result["has_object"][0]
        if not has_object:
            return result

        if raw_image is None or preprocess_multi is None:
            result["note"] = (
                "Binary predicted object present, but raw_image/preprocess_multi not provided; "
                "skipping LayerCAM crop and multiclass prediction."
            )
            return result

        # LayerCAM internally needs gradients; enable them temporarily
        with torch.enable_grad():
            cam = self.layercam.generate(x_bin, class_idx=self.relevant_class)

        pil_img = self._to_pil(raw_image)

        # Preserve float precision when resizing the attribution map.
        cam_np = np.asarray(
            Image.fromarray(cam.astype(np.float32)).resize(
                pil_img.size, resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).clip(0.0, 1.0)
        crop_pil, bbox = crop_from_cam_bbox_adjusted(
            pil_img, cam_np, threshold=self.localization_threshold)
        result["bbox_xyxy"] = list(bbox) if bbox is not None else None
        result["crop_status"] = "ok" if bbox is not None else "empty_cam_full_image"
        result["localization_method"] = "Layer-CAM"
        crop_pil = crop_pil.resize(self.multiclass_input_size, Image.Resampling.BILINEAR)

        x_multi = preprocess_multi(crop_pil)
        if x_multi.ndim == 3:
            x_multi = x_multi.unsqueeze(0)
        if x_multi.ndim != 4 or x_multi.shape[0] != 1:
            raise ValueError("preprocess_multi must return (C,H,W) or (1,C,H,W) tensor.")

        x_multi = x_multi.to(device)

        logits_multi = self.multiclass_model(x_multi)
        probs_multi = torch.softmax(logits_multi, dim=1)
        pred_multi = int(torch.argmax(logits_multi, dim=1).item())
        conf_multi = float(probs_multi[0, pred_multi].item())

        result["multiclass_prediction"] = pred_multi
        result["multiclass_confidence"] = conf_multi

        if self.class_index is not None and 0 <= pred_multi < len(self.class_index):
            info = self.class_index[pred_multi]
            result["name"] = info.name
            result["kind"] = info.kind

        if return_cam:
            result["cam"] = cam_np

        return result

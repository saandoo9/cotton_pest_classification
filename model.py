# dual_stage_model.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from gradcam import GradCAM, crop_from_gradcam_bbox_adjusted


@dataclass(frozen=True)
class ClassInfo:
    """Optional metadata for a multiclass label."""
    name: str
    kind: str


RawImage = Union[np.ndarray, Image.Image]          # OpenCV ndarray (H,W,C) or PIL Image
PreprocessFn = Callable[[Image.Image], torch.Tensor]  # user preprocessing for multiclass crop


class DualStageCropClassifier(nn.Module):
    """
    Dual-stage model with fixed (hardcoded) checkpoints:
      - Binary checkpoint:  modelo_algodon.pth
      - Multiclass checkpoint: best_model_aux_test_1.pth

    Preprocessing stays outside the model:
      - fit() expects preprocessed tensors from the user.
      - predict() expects x_bin as a preprocessed tensor and optionally raw_image + preprocess_multi.
    """

    
    BINARY_CKPT_PATH = "binary_classifier.pth"
    MULTICLASS_CKPT_PATH = "multiclass_classifier.pth"
    # Default GradCAM target layer path for the binary model (string path resolver)
    BINARY_TARGET_LAYER_PATH = "features.-1"


    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        gradcam_threshold: float = 0.4,
        multiclass_input_size: Tuple[int, int] = (224, 224),
        class_index: Optional[Sequence[Dict[str, str]]] = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradcam_threshold = float(gradcam_threshold)
        self.multiclass_input_size = tuple(multiclass_input_size)

        # Hardcoded model loading
        self.binary_model = self._load_model(self.BINARY_CKPT_PATH)
        self.multiclass_model = self._load_model(self.MULTICLASS_CKPT_PATH)

        # Hardcoded target layer path (can be edited in the constants above)
        target_layer = self._resolve_layer(self.binary_model, self.BINARY_TARGET_LAYER_PATH)
        self.gradcam = GradCAM(self.binary_model, target_layer)

        self.class_index: Optional[List[ClassInfo]] = None
        if class_index is not None:
            self.set_class_index(class_index)

    # ----------------------------
    # Utilities
    # ----------------------------

    def _load_model(self, path: str) -> nn.Module:
        """Load a torch model checkpoint, move it to device, set eval by default."""
        model = torch.load(path, map_location=self.device, weights_only = False)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Loaded object from {path} is not a torch.nn.Module.")
        model.to(self.device)
        model.eval()
        return model

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
        Differentiable forward for training (no GradCAM cropping here).
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
        binary_no_object_class: int = 0,
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

        x_bin = x_bin.to(device)

        logits_bin = self.binary_model(x_bin)
        probs_bin = torch.softmax(logits_bin, dim=1)
        pred_bin = torch.argmax(logits_bin, dim=1)  # (B,)
        conf_bin = probs_bin.gather(1, pred_bin.view(-1, 1)).squeeze(1)  # (B,)

        result: Dict[str, Any] = {
            "binary_prediction": pred_bin.detach().cpu().tolist(),
            "binary_confidence": conf_bin.detach().cpu().tolist(),
            "has_object": [int(p.item()) != int(binary_no_object_class) for p in pred_bin],
        }

        # Only single-image mode supports GradCAM crop here
        if x_bin.shape[0] != 1:
            return result

        has_object = result["has_object"][0]
        if not has_object:
            return result

        if raw_image is None or preprocess_multi is None:
            result["note"] = (
                "Binary predicted object present, but raw_image/preprocess_multi not provided; "
                "skipping GradCAM crop and multiclass prediction."
            )
            return result

        # GradCAM internally needs gradients; enable them temporarily
        with torch.enable_grad():
            cam = self.gradcam.generate(x_bin)

        pil_img = self._to_pil(raw_image)

        cam_resized = (
            Image.fromarray((cam * 255).astype(np.uint8))
            .resize(pil_img.size, resample=Image.Resampling.BILINEAR)
        )
        cam_np = (np.array(cam_resized, dtype=np.float32) / 255.0).clip(0.0, 1.0)

        try:
            crop_pil = crop_from_gradcam_bbox_adjusted(
                pil_img,
                cam_np,
                threshold=self.gradcam_threshold,
            ).resize(self.multiclass_input_size)
        except Exception:
            crop_pil = pil_img.resize(self.multiclass_input_size)

        x_multi = preprocess_multi(crop_pil)
        if x_multi.ndim == 3:
            x_multi = x_multi.unsqueeze(0)
        if x_multi.ndim != 4:
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
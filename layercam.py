"""Single-layer Layer-CAM and square cropping for the binary relevance filter."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class LayerCAM:
    """Weight each activation by its positive, spatially varying class gradient.

    A temporary forward hook captures the target feature map. No backward hooks
    or parameter .grad buffers are needed, including with frozen checkpoints.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor, class_idx=1):
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Layer-CAM requires one image with shape (1,C,H,W).")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Layer-CAM needs autograd; call outside torch.inference_mode().")
        activations = []

        def capture(module, inputs, output):
            # Clone so a later in-place operation cannot overwrite the feature map.
            features = output.clone()
            activations.append(features)
            return features

        handle = self.target_layer.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                x = input_tensor.detach().clone().requires_grad_(True)
                logits = self.model(x)
                if len(activations) != 1 or activations[0].ndim != 4:
                    raise RuntimeError("Target layer must produce one spatial feature map per forward.")
                if not 0 <= class_idx < logits.shape[1]:
                    raise ValueError("class_idx is outside the model's output range.")
                features = activations[0]
                gradients, = torch.autograd.grad(logits[0, class_idx], features)
                cam = F.relu((F.relu(gradients) * features).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam[0, 0].detach().cpu().numpy()
                return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            handle.remove()


def get_bbox_from_cam(cam_np, threshold):
    """Return an exclusive-right/bottom box enclosing activations above threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("CAM threshold must lie in [0,1].")
    cam_np = np.asarray(cam_np)
    if cam_np.ndim != 2 or not np.isfinite(cam_np).all():
        raise ValueError("CAM must be a finite 2D array.")
    y, x = np.where(cam_np > threshold)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def crop_from_cam_bbox_adjusted(img_pil, cam_np, threshold):
    """Expand the active box to a square, shift inside the image, then pad.

    Returns (PIL crop, source-image box). The box excludes any black padding.
    An empty map returns the complete image with an explicit None box.
    """
    if np.asarray(cam_np).shape != (img_pil.height, img_pil.width):
        raise ValueError("Resize the CAM to the source image before cropping.")
    bbox = get_bbox_from_cam(cam_np, threshold)
    if bbox is None:
        return img_pil.copy(), None
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1)
    left = (x1 + x2 - side) // 2
    top = (y1 + y2 - side) // 2
    left = max(0, min(left, max(0, img_pil.width - side)))
    top = max(0, min(top, max(0, img_pil.height - side)))
    right = min(left + side, img_pil.width)
    bottom = min(top + side, img_pil.height)
    box = (left, top, right, bottom)
    square = Image.new(img_pil.mode, (side, side), color=0)
    square.paste(img_pil.crop(box), ((side - (right - left)) // 2,
                                    (side - (bottom - top)) // 2))
    return square, box

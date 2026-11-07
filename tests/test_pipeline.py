"""CPU smoke checks without downloading weights or requiring research data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from architectures import build_binary, build_multiclass
from layercam import LayerCAM, crop_from_cam_bbox_adjusted, get_bbox_from_cam
from model import DualStageCropClassifier


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def binary(self, accept=True):
        model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(2, 2))
        with torch.no_grad():
            model[0].weight.fill_(0.5)
            model[0].bias.zero_()
            model[-1].weight.copy_(torch.tensor([[-1., -1.], [1., 1.]]))
            model[-1].bias.copy_(torch.tensor([0., 2. if accept else -20.]))
        return model

    def test_spatial_positive_gradient_formula_and_cleanup(self):
        # Linear spatial weights give a known map; negative gradients must vanish.
        features = nn.Conv2d(1, 1, 1, bias=False)
        head = nn.Linear(4, 2, bias=False)
        model = nn.Sequential(features, nn.Flatten(), head).eval()
        with torch.no_grad():
            features.weight.fill_(1)
            head.weight[1].copy_(torch.tensor([1., -2., 3., 0.]))
        model.requires_grad_(False)
        x = torch.tensor([[[[1., 2.], [3., 4.]]]])
        actual = LayerCAM(model, features).generate(x)
        np.testing.assert_allclose(actual, [[1/9, 0], [1, 0]], atol=1e-6)
        self.assertEqual(len(features._forward_hooks), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_crop_boundaries_empty_and_padding(self):
        image = Image.new("RGB", (8, 4), "white")
        cam = np.zeros((4, 8)); cam[0, 0] = 1
        self.assertEqual(get_bbox_from_cam(cam, .5), (0, 0, 1, 1))
        crop, box = crop_from_cam_bbox_adjusted(image, cam, .5)
        self.assertEqual(crop.size, (1, 1))
        cam[:] = 1
        crop, box = crop_from_cam_bbox_adjusted(image, cam, .5)
        self.assertEqual(crop.size, (8, 8))
        self.assertEqual(box, (0, 0, 8, 4))
        crop, box = crop_from_cam_bbox_adjusted(image, cam * 0, .5)
        self.assertEqual(crop.size, image.size)
        self.assertIsNone(box)

    def test_accepted_frozen_and_rejected_routes(self):
        for accept in (True, False):
            binary = self.binary(accept).requires_grad_(False)
            multi = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 19))
            cascade = DualStageCropClassifier(binary_model=binary, multiclass_model=multi,
                                              localization_threshold=.275,
                                              target_layer_path="0", device="cpu")
            out = cascade.predict(torch.ones(1, 3, 8, 8),
                                  raw_image=Image.new("RGB", (20, 10)),
                                  preprocess_multi=lambda _: torch.ones(3, 8, 8), return_cam=True)
            self.assertEqual(out["has_object"], [accept])
            self.assertEqual("multiclass_prediction" in out, accept)
            if accept:
                self.assertEqual(out["cam"].shape, (10, 20))
                self.assertEqual(out["localization_method"], "Layer-CAM")

    def test_state_dict_loading(self):
        binary = self.binary()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.pth"
            torch.save({"model_state_dict": binary.state_dict()}, path)
            cascade = DualStageCropClassifier(binary_model=self.binary(), binary_checkpoint=str(path),
                                              multiclass_model=nn.Identity(), target_layer_path="0",
                                              localization_threshold=.275, device="cpu")
            for a, b in zip(binary.parameters(), cascade.binary_model.parameters()):
                torch.testing.assert_close(a, b)

    def test_reference_architectures_and_frozen_attribution(self):
        binary, multi = build_binary().eval(), build_multiclass().eval()
        self.assertEqual(sum(p.numel() for p in binary.parameters()), 1821538)
        self.assertEqual(sum(p.numel() for p in multi.parameters()), 6846003)
        binary.requires_grad_(False)
        cam = LayerCAM(binary, binary.features[-1]).generate(torch.rand(1, 3, 128, 128))
        self.assertEqual(cam.shape, (128, 128))
        self.assertTrue(np.isfinite(cam).all())
        with torch.no_grad():
            self.assertEqual(multi(torch.zeros(1, 3, 224, 224)).shape, (1, 19))


if __name__ == "__main__":
    unittest.main()

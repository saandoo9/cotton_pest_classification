"""Reference inference architectures; checkpoint key layouts must match exactly."""

from torch import nn
from torchvision.models import efficientnet_b1, mobilenet_v3_small


def build_binary():
    """MobileNetV3-Small with the 576->1024->256->128->64->2 head."""
    model = mobilenet_v3_small(weights=None)
    layers = []
    widths = (576, 1024, 256, 128, 64, 2)
    for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(inputs, outputs))
        if index < len(widths) - 2:
            layers.extend((nn.Hardswish(), nn.Dropout(p=0.2)))
    model.classifier = nn.Sequential(*layers)
    return model


def build_multiclass():
    """EfficientNet-B1 with the 1280->256->19 head (6,846,003 parameters)."""
    model = efficientnet_b1(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(1280, 256), nn.ReLU(),
        nn.Dropout(p=0.2), nn.Linear(256, 19),
    )
    return model

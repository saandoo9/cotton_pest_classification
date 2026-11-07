"""Run the relevance-filter -> Layer-CAM -> multiclass cascade on one image."""

import argparse
import importlib
import json

from PIL import Image

from model import DualStageCropClassifier
from utils import prepare_image, prepare_image_bin


def load_factory(spec):
    """Import a user-provided no-argument architecture factory."""
    if spec is None:
        return None
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("Architecture factories use module:function syntax.")
    return getattr(importlib.import_module(module), name)()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--binary-checkpoint", required=True)
    parser.add_argument("--multiclass-checkpoint", required=True)
    parser.add_argument("--threshold", required=True, type=float,
                        help="Layer-CAM threshold selected on validation for this checkpoint")
    parser.add_argument("--binary-factory", help="Architecture factory module:function for a state dict")
    parser.add_argument("--multiclass-factory", help="Architecture factory module:function for a state dict")
    parser.add_argument("--trust-checkpoints", action="store_true",
                        help="Allow loading trusted full-model pickle files")
    parser.add_argument("--target-layer", default="features.-1")
    parser.add_argument("--relevant-class", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default=None, help="For example cpu or cuda:0")
    parser.add_argument("--class-index", help="JSON list of name/kind metadata in training label order")
    parser.add_argument("--cam-output", help="Optional grayscale attribution PNG path")
    args = parser.parse_args()
    class_index = None
    if args.class_index:
        with open(args.class_index, encoding="utf-8") as stream:
            class_index = json.load(stream)
    model = DualStageCropClassifier(
        device=args.device,
        localization_threshold=args.threshold,
        binary_checkpoint=args.binary_checkpoint,
        multiclass_checkpoint=args.multiclass_checkpoint,
        binary_model=load_factory(args.binary_factory),
        multiclass_model=load_factory(args.multiclass_factory),
        trust_checkpoint=args.trust_checkpoints,
        target_layer_path=args.target_layer,
        relevant_class=args.relevant_class,
        class_index=class_index,
    )
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    result = model.predict(prepare_image_bin(image), raw_image=image,
                           preprocess_multi=prepare_image, return_cam=bool(args.cam_output))
    cam = result.pop("cam", None)
    if cam is not None:
        Image.fromarray((cam * 255).round().astype("uint8")).save(args.cam_output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

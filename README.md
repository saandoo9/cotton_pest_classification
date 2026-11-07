# Cotton insect classification with Layer-CAM

A PyTorch cascade for identifying 19 insect categories relevant to cotton crop
monitoring, including pests and beneficial insects:

1. A binary filter predicts whether an image is relevant to the target task.
2. For accepted images, **Layer-CAM** generates an activation map for the relevant
   class from the binary model's final feature layer (`features[-1]`). Thresholding
   defines a bounding box, expanded to a square crop.
3. A separate multiclass classifier assigns a target-category label to the crop.

The study's reference architectures are **MobileNetV3-Small** for filtering and
**EfficientNet-B1** with a 19-output head for classification. The wrapper accepts
other compatible models. The binary filter predicts relevance; it does not detect
individual objects. Each accepted image produces one crop and one target label.

## Files

| File | Purpose |
| --- | --- |
| `layercam.py` | Single-layer attribution, thresholding and square cropping |
| `model.py` | `DualStageCropClassifier`, prediction and basic training helper |
| `pipeline.py` | Command-line inference on one image |
| `architectures.py` | Reference MobileNetV3-Small and EfficientNet-B1 factories |
| `utils.py` | RGB preprocessing for both classifiers |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

## Installation

Use Python 3.9 or later and a compatible PyTorch/torchvision pair.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
OpenCV is optional: the CLI uses Pillow. Raw NumPy images passed to the wrapper
are interpreted as OpenCV BGR/BGRA. The preprocessing helpers expect Pillow or
RGB NumPy images and convert to RGB before resizing.

## Checkpoints and threshold

**Trained checkpoints are available as releases.** Supply a binary model and
a matching multiclass model, with training-time preprocessing and label order.
The supplied helpers use ImageNet normalization and resize to 128 × 128 for the
binary model and 224 × 224 for the multiclass model. The cotton taxa dataset is 
available at https://zenodo.org/records/17579307 and the general insect dataset
at https://zenodo.org/records/8325384

Two checkpoint formats are supported:

- **State dictionaries**: supply a matching architecture. Plain state dictionaries
  and dictionaries containing `state_dict` or `model_state_dict` load strictly.
  The architecture must have the same head and parameter names as during training.
- **Complete serialized `nn.Module` objects**: opt in with `--trust-checkpoints`
  or `trust_checkpoint=True`. Only load trusted files: pickle can execute code.
  Original custom model classes must remain importable.

The **Layer-CAM threshold is required**. Select it using downstream validation
macro-F1 for the corresponding checkpoint; do not tune it on test images. In the
study, repeats 1–5 selected 0.350, 0.350, 0.275, 0.275 and 0.275, respectively.
These values are checkpoint-specific. Repeat 3 was selected by validation macro-F1.

## Command-line pipeline

For trusted full-model checkpoints:

```bash
python pipeline.py image.jpg \
  --binary-checkpoint binary_classifier.pth \
  --multiclass-checkpoint multiclass_classifier.pth \
  --threshold 0.275 \
  --trust-checkpoints \
  --cam-output layercam.png
```

The example threshold applies only to a matching validation-selected checkpoint.
Binary labels default to 0 = irrelevant and 1 = relevant. Use `--relevant-class 0`
for reversed labels, `--device cpu` to force CPU, and `--target-layer features.-1`
to specify a different target layer.

For state dictionaries, the supplied reference factories can be used when their
module/key layout matches your checkpoints:

```bash
python pipeline.py image.jpg \
  --binary-checkpoint binary_state_dict.pth \
  --multiclass-checkpoint b1_state_dict.pth \
  --binary-factory architectures:build_binary \
  --multiclass-factory architectures:build_multiclass \
  --threshold 0.275
```

`architectures.py` constructs models without downloading pretrained weights. Its
heads have 576→1024→256→128→64→2 (binary) and 1280→256→19 (multiclass) units,
with 1,821,538 and 6,846,003 total parameters respectively. Different module ordering
can change checkpoint keys even with identical parameter counts. If loading fails,
provide factories from your exact training implementation using the same
`module:function` syntax. The supplied factories are inference templates; historical
training settings, including dropout, must be checked against the experiment code.

The CLI prints JSON. Rejected images return binary results only. Accepted images
also return `multiclass_prediction`, `multiclass_confidence`, `localization_method`,
`bbox_xyxy` and `crop_status`. Boxes use original-image coordinates with exclusive
right/bottom bounds. Empty masks fall back to the full image, reporting
`crop_status="empty_cam_full_image"` and `bbox_xyxy=null`. Other errors are raised.

Optionally pass `--class-index classes.json`: a list of objects with `name` and
`kind` fields in the checkpoint's class-index order (`kind` can be `Pest` or
`Beneficial`). Otherwise, predictions use numeric labels.

## Python API

```python
from PIL import Image
from model import DualStageCropClassifier
from utils import prepare_image_bin, prepare_image

# Supply your constructed/loaded nn.Module objects.
cascade = DualStageCropClassifier(
    binary_model=binary_model,
    multiclass_model=multiclass_model,
    localization_threshold=0.275,
)
image = Image.open("image.jpg").convert("RGB")
result = cascade.predict(
    prepare_image_bin(image),
    raw_image=image,
    preprocess_multi=prepare_image,
    return_cam=True,
)
```

When present, `result["cam"]` is a normalized floating-point map aligned with the
source image. Layer-CAM weights each activation by its positive, spatially varying
gradient of the **pre-softmax relevant-class score**, then applies ReLU to the
channel sum. Only the selected feature layer is used; there is no multi-layer fusion.

Frozen checkpoints are supported by enabling gradients on the attribution input.
Do not wrap the pipeline in `torch.inference_mode()`: attribution needs autograd.
Binary prediction supports batches; localization and multiclass prediction support
one image at a time. Instances are intended for serial use.

## Training and reproduction

`forward(x_bin, x_multi)` returns logits. The existing `fit()` helper trains
either or both models on preprocessed batches, using `x_bin`/`y_bin` and/or
`x_multi`/`y_multi`. The caller must prepare training crops and augmentation;
localization is not differentiable through the pipeline.

The helper does not implement the paper's scheduler, validation-selected
checkpoints, weighted losses, development splits or separate backbone/head
learning rates. Full reproduction requires the original experiment scripts,
data and matching checkpoints. This repository provides the inference components
and a basic training interface, not the complete experiment archive.

The updated crop helper keeps CAM values as floats during resizing and uses
exclusive pixel bounds so single-pixel regions remain valid. These details differ
from the former uint8 Grad-CAM helper. Exact reproduction of archived predictions
requires checking the original crop implementation as well as weights and thresholds.

Five development repeats in the paper share the same fixed 42-image relevant
test set; they are not five independent test populations. Tesla T4 timings do
not establish edge-device performance, and field deployment has not been validated.

CPU checks without research data or downloaded weights:

```bash
python -m unittest discover -s tests -v
```

## Reference and license

Jiang et al., “LayerCAM: Exploring Hierarchical Class Activation Maps for
Localization,” IEEE Transactions on Image Processing, 2021.
[DOI: 10.1109/TIP.2021.3089943](https://doi.org/10.1109/TIP.2021.3089943).

Code is licensed under [MIT](LICENSE). Dataset, image, dependency and pretrained
weight licenses remain those of their respective owners.

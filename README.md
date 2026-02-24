# Dual Stage Crop Classifier

A PyTorch-based **two-stage classification pipeline** with
GradCAM-guided cropping.

The model performs:

-   **Binary classification** → detect object presence (e.g., insect vs
    no insect)\
-   **GradCAM-based ROI extraction** (if object detected)\
-   **Multiclass classification** on the cropped region

The system is designed to be:

-   Fully retrainable\
-   Modular\
-   Preprocessing-agnostic (handled outside the model)\
-   Compatible with OpenCV or PIL images for inference\
-   Independent from any API framework

------------------------------------------------------------------------

# Architecture Overview

    Raw Image
        ↓
    User Preprocessing → x_bin (tensor)
        ↓
    Binary Model
        ↓
    If positive:
        GradCAM
        ↓
    ROI Crop
        ↓
    User Preprocessing → x_multi (tensor)
        ↓
    Multiclass Model

### Important design decision

> Image preprocessing is handled outside the model.\
> The model only consumes tensors.

------------------------------------------------------------------------

# Project Structure

    dual_stage_model.py      # Main retrainable model class
    gradcam.py               # GradCAM implementation
    utils.py                 # User preprocessing utilities

------------------------------------------------------------------------

# Requirements

    torch>=2.0.0
    torchvision>=0.15.0
    numpy>=1.23.0
    Pillow>=9.0.0
    opencv-python>=4.7.0

Install dependencies:

``` bash
pip install -r requirements.txt
```

If using CUDA, install torch manually from:\
https://pytorch.org/get-started/locally/

------------------------------------------------------------------------

# Model Behavior

The class loads two **hardcoded checkpoints**:

    modelo_algodon.pth
    best_model_aux_test_1.pth

You can change them inside the class:

``` python
DualStageCropClassifier.BINARY_CKPT_PATH
DualStageCropClassifier.MULTICLASS_CKPT_PATH
```

The GradCAM target layer is resolved from:

``` python
BINARY_TARGET_LAYER_PATH = "features.-1"
```

Modify this if your binary model architecture differs.

------------------------------------------------------------------------

# Usage

## Initialize Model

``` python
from dual_stage_model import DualStageCropClassifier

model = DualStageCropClassifier()
```

------------------------------------------------------------------------

## Training

`fit()` trains binary and/or multiclass models using **preprocessed
tensors**.

### Expected batch format

``` python
{
    "x_bin": tensor (B,C,H,W),
    "y_bin": tensor (B,),
    "x_multi": tensor (B,C,H,W),
    "y_multi": tensor (B,)
}
```

### Example Training

``` python
history = model.fit(
    dataloader=train_loader,
    epochs=10,
    criterion_binary=torch.nn.CrossEntropyLoss(),
    criterion_multiclass=torch.nn.CrossEntropyLoss(),
    train_binary=True,
    train_multiclass=True,
    lr_binary=1e-4,
    lr_multiclass=5e-5,
)
```

Binary and multiclass use separate parameter groups internally.

------------------------------------------------------------------------

## Inference

Binary preprocessing must be done by the user.

### Example with OpenCV

``` python
import cv2
from utils import prepare_image_bin, prepare_image

raw = cv2.imread("image.jpg")  # BGR ndarray

x_bin = prepare_image_bin(...)  # your preprocessing → tensor (1,C,H,W)

out = model.predict(
    x_bin,
    raw_image=raw,
    preprocess_multi=lambda pil: prepare_image(pil)
)

print(out)
```

------------------------------------------------------------------------

# Predict Return Format

### Binary only (batch supported)

``` python
{
  "binary_prediction": [0],
  "binary_confidence": [0.98],
  "has_object": [False]
}
```

### Full pipeline (single image)

``` python
{
  "binary_prediction": [1],
  "binary_confidence": [0.99],
  "has_object": [True],
  "multiclass_prediction": 7,
  "multiclass_confidence": 0.92,
  "name": "Aphididae",
  "kind": "Pest"
}
```

------------------------------------------------------------------------

# Important Design Notes

-   GradCAM cropping is non-differentiable and not used during training.
-   Provide ground-truth crops for multiclass training.
-   Binary supports batch inference.
-   GradCAM + crop runs only in single-image mode.

------------------------------------------------------------------------

# Extending to Other Domains

To adapt for another crop or object:

1.  Replace training dataset
2.  Train binary model
3.  Train multiclass model
4.  Save checkpoints with same filenames or update class constants

No architectural changes required.


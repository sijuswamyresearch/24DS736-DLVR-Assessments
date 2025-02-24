# Multi-Class Object Detection Model Documentation

## Overview

This project implements a multi-class object detection model using the **MobileNetV2** architecture. The model is designed to classify objects into one of three classes (`butterfly`, `dalmatian`, `dolphin`) and predict their bounding box coordinates. The model uses transfer learning by leveraging the pre-trained MobileNetV2 as the base network, followed by custom heads for classification and bounding box regression.

---

## Table of Contents

1. [Dependencies](#dependencies)
2. [Global Settings](#global-settings)
3. [Dataset Loading and Preprocessing](#dataset-loading-and-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Inference and Testing](#inference-and-testing)
8. [Saving and Loading the Model](#saving-and-loading-the-model)
9. [Results Visualization](#results-visualization)

---

## Dependencies

The following libraries are required to run the code:

- `imutils`
- `os`
- `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `sklearn`
- `tensorflow`
- `pickle`

You can install the necessary Python packages using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow opencv-python imutils
```

## Global Settings

The following global settings are defined at the beginning of the script:

- `TARGET_IMAGE_SIZE`: The input size for MobileNetV2, which expects images of size (224, 224).
- `INPUT_SHAPE`: The shape of the input tensor, including the number of channels (224, 224, 3).
- `BATCH_SIZE`: The batch size used during training.
- `NUM_EPOCHS`: The number of epochs for training.
- `INIT_LR`: The initial learning rate for the Adam optimizer.
- `CLASSES`: The list of classes for multi-class classification (butterfly, dalmatian, dolphin).

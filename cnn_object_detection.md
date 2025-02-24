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
10. [Conclusion](#conclusion)

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

## Dataset Loading and Preprocessing
>**Steps:**

*Annotation Parsing :*

- The annotations are read from text files located in the directory specified by ann_path. Each row contains the filename and bounding box coordinates.
- The bounding box coordinates are normalized relative to the image dimensions.
  
*Image Loading and Resizing :*

- Images are loaded using `cv2.imread()` and resized to the target size (224, 224) using `load_img()`.
- Images are preprocessed using `MobileNetV2'`s preprocessing function (`preprocess_input`).
  
*Data Storage :*

- Images, labels, bounding boxes, and image paths are stored in separate lists and later converted to NumPy arrays.
- Labels are one-hot encoded using LabelBinarizer.
  
*Data Splitting :*
The dataset is split into training and testing sets using an 80/20 split with train_test_split.

## Model Architecture

- **MobileNetV2** : The pre-trained `MobileNetV2` model is used as the base network. The top layers are removed (`include_top=False`), and the base model is frozen (`baseModel.trainable = False`).
- *Custom Heads:*
1. Classification Head :
   
- A fully connected layer with 128 neurons and ReLU activation.
- Dropout layer with a dropout rate of 0.5.
- Output layer with `softmax` activation for multi-class classification.
  
2. Bounding Box Regression Head :
  
- Fully connected layers with ReLU activation.
- Output layer with `sigmoid` activation to predict normalized bounding box coordinates.
  
## Model Summary:

The model outputs two predictions:

- class_label: The predicted class label.
- bounding_box: The predicted bounding box coordinates.

## Model Training

**Compilation:**
- *Optimizer :* Adam optimizer with an initial learning rate of 1e-4.
- *Loss Functions :*
  - categorical_crossentropy for classification.
  - mean_squared_error for bounding box regression.
- *Metrics :*
  - Accuracy for classification.
  - Mean Absolute Error (MAE) for bounding box regression.
- *Loss Weights :*
  - Equal weights are assigned to both classification and bounding box regression (1.0 each).
- *Training:*
  - The model is trained for 25 epochs with a batch size of 32.
  - Validation data is used to monitor performance on unseen data.

## Model Evaluation
After training, the model's performance can be evaluated using metrics such as:

- *Accuracy* : For classification.
- *Mean Average Precision (mAP)* : For object detection tasks (not implemented in this script but can be added).

## Saving and Loading the Model
- **Saving:**
  - The trained model is saved as `model_bbox_regression_and_classification_mobi.h5`.
  - The label binarizer (lb) is saved as a pickle file (lb.pickle).
- **Loading:**
  - The model and label binarizer can be loaded using `load_model()` and `pickle.loads()`.
 
## Inference and Testing
>Steps:

*Load Test Image :*
- Test images are loaded and resized to (224, 224) using `load_img()` and preprocessed with `preprocess_input`.
- 
*Predictions :*
- The model predicts both the class label and bounding box coordinates.
- Bounding box coordinates are scaled back to the original image dimensions.
  
*Visualization :*

- The predicted bounding box and class label are drawn on the original image using OpenCV functions (`cv2.rectangle` and `cv2.putText`).
- The result is displayed using `matplotlib`.

## Results Visualization

The results of the model are visualized by drawing the predicted bounding box and class label on the test images. The visualization is done using matplotlib.

## Example Usage
To run inference on a new set of images:

1. Place the test images in the appropriate directory.
2. Update the testing_multiclass.txt file with the paths to the test images.
3. Run the script to load the model, perform inference, and visualize the results.

## Notes
- Data Augmentation : Consider adding data augmentation techniques (e.g., rotation, flipping) to improve model robustness.
- Hyperparameter Tuning : Experiment with different learning rates, batch sizes, and loss weights to optimize performance.
- Evaluation Metrics : Implement additional evaluation metrics like `mAP` for a more comprehensive assessment of the model's performance.

## Conclusion
This multi-class object detection model exploits the power of `MobileNetV2` for efficient feature extraction and adds custom heads for classification and bounding box regression. The model is suitable for small-scale object detection tasks and can be further improved with additional tuning and evaluation.

Link to the code: [object detection](https://github.com/sijuswamyresearch/24DS736-DLVR-Assessments/blob/main/Multi-object-detection-and-bounding-box-regression.py)

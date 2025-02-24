# RCNN Object Detection Inference Documentation

This document provides a detailed explanation of the inference pipeline for an object detection model based on the **Region-based Convolutional Neural Network (RCNN)** approach. The pipeline includes **Mean Average Precision (mAP)** and **Average Precision (AP) per class** calculations to evaluate the model's performance.

---

## Table of Contents

1. [Dependencies](#dependencies)
2. [Global Settings](#global-settings)
3. [Intersection over Union (IoU) Calculation](#intersection-over-union-iou-calculation)
4. [Image Resizing with Padding](#image-resizing-with-padding)
5. [Bounding Box Scaling](#bounding-box-scaling)
6. [Bounding Box Validation](#bounding-box-validation)
7. [mAP and AP Per Class Calculation](#map-and-ap-per-class-calculation)
8. [Inference Function](#inference-function)
9. [Main Execution](#main-execution)

---

## Dependencies

The following libraries are required to run the code:

- `os`
- `cv2` (OpenCV)
- `numpy`
- `sklearn`
- `tensorflow.keras`
- `pickle`

Install the necessary Python packages using pip:

```bash
pip install numpy scikit-learn tensorflow opencv-python
```

## Global Settings

The global settings define key parameters for the inference pipeline:

- **TARGET_IMAGE_SIZE**: Input size for MobileNetV2 (224, 224)
- **IOU_THRESHOLD_POSITIVE**: IoU threshold for positive samples (0.3)
- **IOU_THRESHOLD_NEGATIVE**: IoU threshold for negative samples (0.2)
- **IMAGE_DIRECTORY**: Directory containing the dataset
- **ANNOTATION_FILE_PREFIX**: Prefix for annotation files
- **CLASSES**: List of classes (butterfly, dalmatian, dolphin)
- **MAX_IMAGES_PER_CLASS**: Number of images processed per class (30)
- **MAX_PROPOSALS**: Maximum number of region proposals generated per image (10)
- **MODEL_PATH**: Path to the pre-trained model file

---

## Intersection over Union (IoU) Calculation

**Function**: `get_iou(bb1, bb2)`

**Purpose**:  
Calculates the Intersection over Union (IoU) between two bounding boxes.

**Inputs**:
- `bb1`, `bb2`: Dictionaries containing bounding box coordinates (`x1`, `y1`, `x2`, `y2`).

**Output**:
- IoU value (*float*).

---

## Image Resizing with Padding

**Function**: `resize_image_with_padding(image, target_size)`

**Purpose**:  
Resizes an image while preserving its aspect ratio by adding padding.

**Steps**:
1. **Calculate the scaling factor**  
   Determine the scaling factor based on the target size.
2. **Resize the image**  
   Resize the image using the calculated scaling factor.
3. **Add padding**  
   Add padding to make the image dimensions match the target size.

**Output**:
- Resized and padded image.
- Scaling factor.
- Padding values.

---

## Bounding Box Scaling

**Function**: `scale_bounding_box(bbox, scale, padding)`

**Purpose**:  
Scales bounding box coordinates after resizing and padding.

**Inputs**:
- `bbox`: Original bounding box coordinates (`x1`, `y1`, `x2`, `y2`).
- `scale`: Scaling factor applied during resizing.
- `padding`: Padding values added during resizing.

**Output**:
- Scaled bounding box coordinates.

---

## Bounding Box Validation

**Function**: `validate_bbox(bbox, target_size)`

**Purpose**:  
Validates a bounding box to ensure it has valid dimensions.

**Inputs**:
- `bbox`: Bounding box coordinates (`x1`, `y1`, `x2`, `y2`).
- `target_size`: Target image size (`width`, `height`).

**Output**:
- Boolean indicating whether the bounding box is valid.

---

## mAP and AP Per Class Calculation

**Function**: `compute_map(all_predictions, all_ground_truths, iou_threshold=0.5)`

**Purpose**:  
Computes the Mean Average Precision (mAP) and Average Precision (AP) per class for object detection.

**Steps**:
1. Iterate over each class and filter predictions and ground truths.
2. Sort predictions by confidence score (descending).
3. Compute True Positives (TP) and False Positives (FP) based on IoU thresholds.
4. Calculate Precision and Recall.
5. Compute AP using the trapezoidal rule.
6. Compute mAP as the mean of AP values across all classes.

**Inputs**:
- `all_predictions`: List of predicted bounding boxes, class labels, and confidence scores.
- `all_ground_truths`: List of ground truth bounding boxes and class labels.
- `iou_threshold`: IoU threshold for matching predictions to ground truths.

**Outputs**:
- **mAP**: Mean Average Precision across all classes.
- **ap_per_class**: Dictionary of AP values for each class.

---

## Inference Function

**Function**: `run_inference(image_directory, annotation_file_prefix, model_path, lb)`

**Purpose**:  
Performs inference on all images in the specified directory and computes mAP and AP per class.

**Steps**:
1. **Load and compile the model**:  
   Load the pre-trained model and explicitly compile it.
2. **Process annotations per class**:  
   Iterate over each class and process annotations.
3. **For each image**:
   - Parse the annotation file to extract the ground truth bounding box.
   - Resize the image with padding and scale the bounding box coordinates.
   - Preprocess the image and perform predictions.
   - Decode the predicted class label and normalize the predicted bounding box.
   - Validate both ground truth and predicted bounding boxes.
   - Store predictions and ground truths for evaluation.
4. **Compute mAP and AP per class**:  
   Use the `compute_map` function to compute evaluation metrics.

**Inputs**:
- `image_directory`: Directory containing the images.
- `annotation_file_prefix`: Prefix for annotation files.
- `model_path`: Path to the pre-trained model file.
- `lb`: Label binarizer used for encoding/decoding class labels.

**Outputs**:
- Prints mAP and AP per class to the console.

---

## Main Execution

**Workflow**:
1. Load the label binarizer (`lb`) from the saved pickle file.
2. Call the `run_inference` function to:
   - Perform inference on all images in the dataset.
   - Compute and display mAP and AP per class.

**Example Output**:

```plaintext
[INFO] Running inference...
[INFO] Model loaded and compiled successfully.
[INFO] Computing mAP...
mAP: 0.75
AP per class: {'butterfly': 0.80, 'dalmatian': 0.70, 'dolphin': 0.75}
```

## Notes

- **Error Handling**:  
  The pipeline includes error handling for missing images, invalid bounding boxes, and other potential issues.

- **Relaxed IoU Threshold**:  
  A relaxed IoU threshold of 0.3 is used for mAP calculation to account for small localization errors.

- **Validation**:  
  Both ground truth and predicted bounding boxes are validated to ensure they have valid dimensions before evaluation.

---

## Conclusion

This inference pipeline evaluates the performance of an RCNN-based object detection model using mAP and AP per class metrics. The pipeline ensures robust preprocessing, validation, and evaluation steps to provide meaningful insights into the model's accuracy and localization capabilities.

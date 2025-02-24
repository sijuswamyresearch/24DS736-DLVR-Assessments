# Object Detection Inference Documentation

This document provides a detailed explanation of the inference pipeline for an object detection model. The pipeline uses a pre-trained model (based on MobileNetV2) to predict class labels and bounding boxes for images in a dataset.

---

## Table of Contents

1. [Dependencies](#dependencies)
2. [Global Settings](#global-settings)
3. [Intersection over Union (IoU) Calculation](#intersection-over-union-iou-calculation)
4. [Image Resizing with Padding](#image-resizing-with-padding)
5. [Bounding Box Scaling](#bounding-box-scaling)
6. [Bounding Box Denormalization](#bounding-box-denormalization)
7. [Bounding Box Validation](#bounding-box-validation)
8. [Visualization of Results](#visualization-of-results)
9. [Loading the Model for Inference](#loading-the-model-for-inference)
10. [Running Inference](#running-inference)

---

## Dependencies

The following libraries are required to run the code:

- `os`
- `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `pickle`
- `sklearn`
- `tensorflow`

Install the necessary Python packages using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow opencv-python
```

## Global Settings

The global settings define key parameters for the inference pipeline:

- `TARGET_IMAGE_SIZE`: Input size for MobileNetV2 (224, 224)
- `IOU_THRESHOLD_POSITIVE`: IoU threshold for positive samples (0.7)
- `IOU_THRESHOLD_NEGATIVE`: IoU threshold for negative samples (0.3)
- `IMAGE_DIRECTORY`: Directory containing the dataset
- `ANNOTATION_FILE_PREFIX`: Prefix for annotation files
- `CLASSES`: List of classes (butterfly, dalmatian, dolphin)
- `MAX_IMAGES_PER_CLASS`: Number of images processed per class (30)
- `MAX_PROPOSALS`: Maximum number of region proposals generated per image (10)
- `BATCH_SIZE`: Batch size for training (16)
- `NUM_EPOCHS`: Number of training epochs (10)
- `INIT_LR`: Initial learning rate (3e-5)

---

## Intersection over Union (IoU) Calculation

*Function*: `get_iou(bb1, bb2)`

*Purpose*:  
Calculates the Intersection over Union (IoU) between two bounding boxes.

*Inputs*:
- `bb1`, `bb2`: Dictionaries containing bounding box coordinates (`x1`, `y1`, `x2`, `y2`).

*Output*:
- *IoU value* (*float*).

---

## Image Resizing with Padding

*Function*: `resize_image_with_padding(image, target_size)`

*Purpose*:  
Resizes an image while preserving its aspect ratio by adding padding.

*Steps*:
1. **Calculate the scaling factor**  
   Determine the scaling factor based on the target size.
2. **Resize the image**  
   Resize the image using the calculated scaling factor.
3. **Add padding**  
   Add padding to make the image dimensions match the target size.

*Output*:
- Resized and padded image.
- Scaling factor.
- Padding values.

---

## Bounding Box Scaling

*Function*: `scale_bounding_box(bbox, scale, padding)`

*Purpose*:  
Scales bounding box coordinates after resizing and padding.

*Inputs*:
- `bbox`: Original bounding box coordinates (`x1`, `y1`, `x2`, `y2`).
- `scale`: Scaling factor applied during resizing.
- `padding`: Padding values added during resizing.

*Output*:
- Scaled bounding box coordinates.

---

## Bounding Box Denormalization

*Function*: `denormalize_bbox(bbox, target_size)`

*Purpose*:  
Converts normalized bounding box coordinates back to pixel values.

*Inputs*:
- `bbox`: Normalized bounding box coordinates (`x1`, `y1`, `x2`, `y2`).
- `target_size`: Target image size (`width`, `height`).

*Output*:
- Denormalized bounding box coordinates.

---

## Bounding Box Validation

*Function*: `validate_bbox(bbox, target_size)`

*Purpose*:  
Validates a bounding box to ensure it has valid dimensions.

*Inputs*:
- `bbox`: Bounding box coordinates (`x1`, `y1`, `x2`, `y2`).
- `target_size`: Target image size (`width`, `height`).

*Output*:
- Boolean indicating whether the bounding box is valid.

---

## Visualization of Results

*Function*: `visualize_results(original_image, gt_label, pred_label, gt_bbox, pred_bbox, target_size)`

*Purpose*:  
Visualizes ground truth and predicted bounding boxes on the original image.

*Steps*:
1. **Denormalize bounding box coordinates**  
   Convert the normalized bounding box coordinates to pixel values.
2. **Validate the predicted bounding box**  
   Ensure that the predicted bounding box has valid dimensions.
3. **Draw bounding boxes and labels**  
   Overlay both the ground truth and predicted bounding boxes along with their labels on the image.
4. **Display the result**  
   Use Matplotlib to display the annotated image.

*Inputs*:
- `original_image`: The original image.
- `gt_label`: Ground truth class label.
- `pred_label`: Predicted class label.
- `gt_bbox`: Ground truth bounding box (normalized).
- `pred_bbox`: Predicted bounding box (normalized).
- `target_size`: Target image size (`width`, `height`).

---

## Loading the Model for Inference

*Function*: `load_model_for_inference(model_path)`

*Purpose*:  
Loads the pre-trained model for inference.

*Steps*:
1. **Load the model**  
   Load the model from the specified path.
2. **Compile the model**  
   Explicitly compile the model with the appropriate loss functions, loss weights, and metrics.

*Inputs*:
- `model_path`: Path to the pre-trained model file.

*Output*:
- Compiled model ready for inference.

---

## Running Inference

*Function*: `run_inference(image_directory, annotation_file_prefix, model_path, lb)`

*Purpose*:  
Performs inference on all images in the specified directory.

**Steps**:
1. *Load the pre-trained model*  
   Use `load_model_for_inference` to load the model.
2. *Iterate over each class*  
   Process annotations for each class.
3. *For each image*:
   - Parse the annotation file to extract the ground truth bounding box.
   - Resize the image with padding and scale the bounding box coordinates.
   - Preprocess the image and perform predictions.
   - Decode the predicted class label and normalize the predicted bounding box.
   - Visualize the results using `visualize_results`.

*Inputs*:
- `image_directory`: Directory containing the images.
- `annotation_file_prefix`: Prefix for annotation files.
- `model_path`: Path to the pre-trained model file.
- `lb`: Label binarizer used for encoding/decoding class labels.

---

## Notes

- *Error Handling*:  
  The pipeline includes error handling for missing images, invalid bounding boxes, and other potential issues.
  
- *Visualization*:  
  The visualization function provides meaningful titles and labels for both ground truth and predicted bounding boxes.
  
- *Model Compilation*:  
  The model is explicitly compiled during inference to ensure compatibility with the loss functions and metrics used during training.

---

## Conclusion

This inference pipeline uses a pre-trained MobileNetV2-based model to predict class labels and bounding boxes for object detection tasks. This inference approach includes robust preprocessing, validation, and visualization steps to ensure accurate and interpretable results.

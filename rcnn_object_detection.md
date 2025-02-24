# Object Detection Using RCNN -Documentation

This project implements an object detection pipeline using the **Region-based Convolutional Neural Network (RCNN)** approach. The model leverages `MobileNetV2` as the base network for feature extraction and includes custom heads for classification and bounding box regression.

---

## Table of Contents

1. [Dependencies](#dependencies)
2. [Global Settings](#global-settings)
3. [Intersection over Union (IoU) Calculation](#intersection-over-union-iou-calculation)
4. [Image Resizing with Padding](#image-resizing-with-padding)
5. [Bounding Box Scaling](#bounding-box-scaling)
6. [Processing Images and Generating Region Proposals](#processing-images-and-generating-region-proposals)
7. [Model Architecture](#model-architecture)
8. [Training the Model](#training-the-model)
9. [Plotting Training History](#plotting-training-history)
10. [Saving the Model](#saving-the-model)

---

## Dependencies

The following libraries are required to run the code:

- `os`
- `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `random`
- `pickle`
- `sklearn`
- `tensorflow`

Install the necessary Python packages using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow opencv-python imutils
```

## Global Settings

The global settings define key parameters for the RCNN pipeline:

- `TARGET_IMAGE_SIZE`: Input size for MobileNetV2 (224, 224)
- `IOU_THRESHOLD_POSITIVE`: IoU threshold for positive samples (0.7)
- `IOU_THRESHOLD_NEGATIVE`: IoU threshold for negative samples (0.3)
- `IMAGE_DIRECTORY`: Directory containing the dataset
- `ANNOTATION_FILE`: Path to annotation files
- `CLASSES`: List of classes (butterfly, dalmatian, dolphin)
- `MAX_IMAGES_PER_CLASS`: Number of images processed per class (20)
- `MAX_PROPOSALS`: Maximum number of region proposals generated per image (20)
- `BATCH_SIZE`: Batch size for training (32)
- `NUM_EPOCHS`: Number of training epochs (20)
- `INIT_LR`: Initial learning rate (3e-5)

## Intersection over Union (IoU) Calculation

**Function**: `get_iou(bb1, bb2)`  
**Purpose**: Calculates the Intersection over Union (IoU) between two bounding boxes.  

### Inputs
- `bb1`, `bb2`: Dictionaries containing bounding box coordinates (`x1`, `y1`, `x2`, `y2`).

### Output
- `IoU value` (*float*).

## Image Resizing with Padding

**Function**: `resize_image_with_padding(image, target_size)`

**Purpose**:  
Resizes an image while preserving its aspect ratio by adding padding.

### Steps:
1. **Calculate the scaling factor**  
   Determine the factor by which the image should be resized based on the target size.
2. **Resize the image**  
   Apply the scaling factor to resize the image.
3. **Add padding**  
   Compute and add padding to the resized image so that its dimensions match the target size.

### Output:
- Resized and padded image
- Scaling factor
- Padding values

## Bounding Box Scaling

**Function**: `scale_bounding_box(bbox, scale, padding)`

*Purpose*:  
Scales bounding box coordinates after resizing and padding.

### Inputs:
- **bbox**: Original bounding box coordinates (`x1, y1, x2, y2`).
- **scale**: Scaling factor applied during resizing.
- **padding**: Padding values added during resizing (e.g., a tuple `(pad_x, pad_y)`).

## Output:
- **Scaled bounding box coordinates**: Adjusted coordinates (`x1, y1, x2, y2`) after applying the scale and padding.


import os
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import pickle

# --- Global Settings ---
TARGET_IMAGE_SIZE = (224, 224)  # Input size for MobileNetV2
IOU_THRESHOLD_POSITIVE = 0.3    # IoU threshold for positive samples
IOU_THRESHOLD_NEGATIVE = 0.2    # IoU threshold for negative samples
IMAGE_DIRECTORY = "../content/gdrive/MyDrive/MC_RCNN"
ANNOTATION_FILE_PREFIX = "../content/gdrive/MyDrive/MC_RCNN/res_"
CLASSES = ["butterfly", "dalmatian", "dolphin"]
MAX_IMAGES_PER_CLASS = 30       # Number of images to process per class
MAX_PROPOSALS = 10              # Max number of region proposals per image
MODEL_PATH = "object_detection_model.h5"

# --- Helper Functions ---
def get_iou(bb1, bb2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def resize_image_with_padding(image, target_size):
    """
    Resize an image while preserving the aspect ratio by adding padding.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scaling factor
    scale = min(target_h / h, target_w / w)

    # Resize the image
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Add padding
    pad_h = (target_h - resized.shape[0]) // 2
    pad_w = (target_w - resized.shape[1]) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded, scale, (pad_w, pad_h)


def scale_bounding_box(bbox, scale, padding):
    """
    Scale bounding box coordinates after resizing and padding.
    """
    x1, y1, x2, y2 = bbox
    pad_w, pad_h = padding

    # Scale coordinates
    x1 = int(x1 * scale) + pad_w
    y1 = int(y1 * scale) + pad_h
    x2 = int(x2 * scale) + pad_w
    y2 = int(y2 * scale) + pad_h

    return [x1, y1, x2, y2]


def validate_bbox(bbox, target_size):
    """
    Validate a bounding box to ensure it has valid dimensions.
    """
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > target_size[1] or y2 > target_size[0]:
        return False
    return True


# --- mAP Calculation ---
def compute_map(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Compute mAP (mean Average Precision) for object detection.
    """
    ap_per_class = {}

    # Iterate over each class
    unique_classes = set([gt['class'] for gt in all_ground_truths])
    for cls in unique_classes:
        # Filter predictions and ground truths for the current class
        cls_preds = [pred for pred in all_predictions if pred['class'] == cls]
        cls_gts = [gt for gt in all_ground_truths if gt['class'] == cls]

        # Sort predictions by confidence score (descending)
        cls_preds.sort(key=lambda x: x['confidence'], reverse=True)

        tp = np.zeros(len(cls_preds))  # True positives
        fp = np.zeros(len(cls_preds))  # False positives
        matched_gt = set()

        for i, pred in enumerate(cls_preds):
            max_iou = -1
            best_match_idx = -1

            for j, gt in enumerate(cls_gts):
                if j in matched_gt:
                    continue

                # Compute IoU between predicted and ground truth bounding boxes
                iou = get_iou(
                    {"x1": pred['bbox']['x1'], "y1": pred['bbox']['y1'], "x2": pred['bbox']['x2'], "y2": pred['bbox']['y2']},
                    {"x1": gt['bbox']['x1'], "y1": gt['bbox']['y1'], "x2": gt['bbox']['x2'], "y2": gt['bbox']['y2']}
                )

                if iou > max_iou:
                    max_iou = iou
                    best_match_idx = j

            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_match_idx)
            else:
                fp[i] = 1

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        num_gt = len(cls_gts)
        recall = tp_cumsum / (num_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        # Compute AP using the trapezoidal rule
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p = np.max(precision[recall >= t]) if np.any(recall >= t) else 0
            ap += p / 11
        ap_per_class[cls] = ap

    # Compute mAP
    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class


# --- Inference Function ---
def run_inference(image_directory, annotation_file_prefix, model_path, lb):
    """
    Perform inference on all images in the specified directory and compute mAP.
    """
    print("[INFO] Running inference...")

    # Step 1: Load the model
    model = load_model(model_path, compile=False)

    # Explicitly compile the model
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        optimizer=opt,
        loss={
            "class_label": "categorical_crossentropy",
            "bounding_box": "mse"
        },
        loss_weights={
            "class_label": 1.0,
            "bounding_box": 10.0  # Increased weight for bounding box loss
        },
        metrics={
            "class_label": "accuracy",
            "bounding_box": "mae"
        }
    )
    print("[INFO] Model loaded and compiled successfully.")

    # Initialize lists to store predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    # Iterate over each class
    for cl in CLASSES:
        annotation_path = annotation_file_prefix + cl + ".txt"
        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file not found at {annotation_path}")
            continue

        # Read annotation file
        with open(annotation_path, "r") as f:
            lines = f.readlines()[:MAX_IMAGES_PER_CLASS]  # Process only the first MAX_IMAGES_PER_CLASS images

        # Process each annotation
        for line in lines:
            try:
                # Parse annotation
                parts = line.strip().split(",")
                image_name = parts[0].replace("annotation_", "image_").replace(".mat", ".jpg")
                image_name = image_name.strip('\ufeff').strip()

                # Construct image path
                image_path = os.path.join(image_directory, cl, image_name)
                if not os.path.exists(image_path):
                    print(f"Error: Image not found at {image_path}")
                    continue

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to load image at {image_path}")
                    continue

                # Ground truth bounding box
                x1, y1, x2, y2 = map(int, parts[1:5])
                gt_box = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

                # Resize image with padding
                resized_image, scale, padding = resize_image_with_padding(image, TARGET_IMAGE_SIZE)

                # Scale bounding box coordinates
                scaled_gt_box = scale_bounding_box([x1, y1, x2, y2], scale, padding)

                # Validate ground truth bounding box
                if not validate_bbox(scaled_gt_box, TARGET_IMAGE_SIZE):
                    print(f"Invalid ground truth bounding box detected for {image_name}. Skipping...")
                    continue

                # Preprocess image
                preprocessed_image = preprocess_input(img_to_array(resized_image))
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

                # Predict class label and bounding box
                preds = model.predict(preprocessed_image)
                pred_label = preds[0][0]  # Predicted class probabilities
                pred_bbox = preds[1][0]   # Predicted bounding box coordinates

                # Decode class label
                class_idx = np.argmax(pred_label)
                class_label = lb.classes_[class_idx]
                confidence = pred_label[class_idx]

                # Normalize predicted bounding box
                normalized_pred_box = {
                    'x1': max(0, min(1, pred_bbox[0])),
                    'y1': max(0, min(1, pred_bbox[1])),
                    'x2': max(0, min(1, pred_bbox[2])),
                    'y2': max(0, min(1, pred_bbox[3]))
                }

                # Validate predicted bounding box
                denormalized_pred_box = [
                    normalized_pred_box['x1'] * TARGET_IMAGE_SIZE[1],
                    normalized_pred_box['y1'] * TARGET_IMAGE_SIZE[0],
                    normalized_pred_box['x2'] * TARGET_IMAGE_SIZE[1],
                    normalized_pred_box['y2'] * TARGET_IMAGE_SIZE[0]
                ]
                if not validate_bbox(denormalized_pred_box, TARGET_IMAGE_SIZE):
                    print(f"Invalid predicted bounding box detected for {image_name}. Skipping...")
                    continue

                # Store predictions and ground truths
                all_predictions.append({
                    'class': class_label,
                    'confidence': confidence,
                    'bbox': normalized_pred_box
                })
                all_ground_truths.append({
                    'class': cl,
                    'bbox': {
                        'x1': scaled_gt_box[0] / TARGET_IMAGE_SIZE[1],
                        'y1': scaled_gt_box[1] / TARGET_IMAGE_SIZE[0],
                        'x2': scaled_gt_box[2] / TARGET_IMAGE_SIZE[1],
                        'y2': scaled_gt_box[3] / TARGET_IMAGE_SIZE[0]
                    }
                })

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

    # Compute mAP
    print("[INFO] Computing mAP...")
    mAP, ap_per_class = compute_map(all_predictions, all_ground_truths, iou_threshold=0.3)  # Use relaxed IoU threshold
    print(f"mAP: {mAP}")
    print(f"AP per class: {ap_per_class}")


# --- Main ---
if __name__ == "__main__":
    # Load the label binarizer
    with open("lb.pickle", "rb") as f:
        lb = pickle.load(f)

    # Run inference and compute mAP
    run_inference(IMAGE_DIRECTORY, ANNOTATION_FILE_PREFIX, MODEL_PATH, lb)

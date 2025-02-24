import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping

# --- Global Settings ---
TARGET_IMAGE_SIZE = (224, 224)  # Input size for MobileNetV2
IOU_THRESHOLD_POSITIVE = 0.7    # IoU threshold for positive samples
IOU_THRESHOLD_NEGATIVE = 0.3    # IoU threshold for negative samples
IMAGE_DIRECTORY = "../content/gdrive/MyDrive/MC_RCNN"
ANNOTATION_FILE_PREFIX = "res_"
CLASSES = ["butterfly", "dalmatian", "dolphin"]
MAX_IMAGES_PER_CLASS = 30       # Number of images to process per class
MAX_PROPOSALS = 10              # Max number of region proposals per image
BATCH_SIZE = 16                 # Batch size for training
NUM_EPOCHS = 10                 # Number of training epochs
INIT_LR = 3e-5                  # Initial learning rate (reduced)

# --- IoU Calculation ---
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

# --- Resize Image with Padding ---
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

# --- Scale Bounding Box Coordinates ---
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

# --- Denormalize Bounding Box ---
def denormalize_bbox(bbox, target_size):
    """
    Denormalize bounding box coordinates to pixel values.
    """
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * target_size[1]),
        int(y1 * target_size[0]),
        int(x2 * target_size[1]),
        int(y2 * target_size[0])
    ]

# --- Validate Bounding Box ---
def validate_bbox(bbox, target_size):
    """
    Validate a bounding box to ensure it has valid dimensions.
    """
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > target_size[1] or y2 > target_size[0]:
        return False
    return True

# --- Visualize Results ---
def visualize_results(original_image, gt_bbox, pred_bbox, target_size):
    """
    Visualize ground truth and predicted bounding boxes.
    """
    # Denormalize ground truth bounding box
    gt_bbox_denorm = denormalize_bbox(gt_bbox, target_size)

    # Denormalize predicted bounding box
    pred_bbox_denorm = denormalize_bbox(pred_bbox, target_size)

    # Validate predicted bounding box
    if not validate_bbox(pred_bbox_denorm, target_size):
        print("Invalid bounding box detected. Skipping...")
        return

    # Draw ground truth bounding box (red)
    cv2.rectangle(original_image, (gt_bbox_denorm[0], gt_bbox_denorm[1]), (gt_bbox_denorm[2], gt_bbox_denorm[3]), (0, 0, 255), 2)
    cv2.putText(
        original_image,
        "GT",
        (gt_bbox_denorm[0], gt_bbox_denorm[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )

    # Draw predicted bounding box (green)
    cv2.rectangle(original_image, (pred_bbox_denorm[0], pred_bbox_denorm[1]), (pred_bbox_denorm[2], pred_bbox_denorm[3]), (0, 255, 0), 2)
    cv2.putText(
        original_image,
        "Pred",
        (pred_bbox_denorm[0], pred_bbox_denorm[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth vs Predicted Bounding Box")
    plt.axis("off")
    plt.show()

# --- Build Model ---
def build_model(input_shape, num_classes):
    """
    Build the object detection model using MobileNetV2 as the base network.
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False
    flatten = Flatten()(base_model.output)
    class_head = Dense(128, activation="relu")(flatten)
    class_head = Dense(num_classes, activation="softmax", name="class_label")(class_head)  # Output for 4 classes
    bbox_head = Dense(128, activation="relu")(flatten)
    bbox_head = Dense(64, activation="relu")(bbox_head)
    bbox_head = Dense(32, activation="relu")(bbox_head)
    bbox_head = Dense(4, activation="sigmoid", name="bounding_box")(bbox_head)
    model = Model(inputs=base_model.input, outputs=[class_head, bbox_head])
    return model

# --- Load Model for Inference ---
def load_model_for_inference(model_path):
    """
    Load the pre-trained model for inference.
    """
    print("[INFO] Loading model...")
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
    return model

# --- Run Inference ---
def run_inference(image_directory, annotation_file_prefix, model_path, lb):
    """
    Perform inference on all images in the specified directory.
    """
    print("[INFO] Running inference...")

    # Step 1: Load the model
    model = load_model_for_inference(model_path)

    # Iterate over each class
    for cl in CLASSES:
        annotation_path = annotation_file_prefix + cl + ".txt"
        if not os.path.exists(annotation_path):
            print(f"Error: Annotation file not found at {annotation_path}")
            continue

        # Read annotation file
        with open(annotation_path, "r") as f:
            lines = f.readlines()

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
                gt_box = [x1, y1, x2, y2]

                # Resize image with padding
                resized_image, scale, padding = resize_image_with_padding(image, TARGET_IMAGE_SIZE)

                # Scale bounding box coordinates
                scaled_gt_box = scale_bounding_box(gt_box, scale, padding)

                # Preprocess image
                preprocessed_image = preprocess_input(img_to_array(resized_image))

                # Expand dimensions for batch prediction
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

                # Predict class label and bounding box
                preds = model.predict(preprocessed_image)
                pred_label = preds[0][0]  # Predicted class probabilities
                pred_bbox = preds[1][0]   # Predicted bounding box coordinates

                # Decode class label
                class_idx = np.argmax(pred_label)
                class_label = lb.classes_[class_idx]

                # Normalize ground truth bounding box
                normalized_gt_box = [
                    scaled_gt_box[0] / TARGET_IMAGE_SIZE[1],
                    scaled_gt_box[1] / TARGET_IMAGE_SIZE[0],
                    scaled_gt_box[2] / TARGET_IMAGE_SIZE[1],
                    scaled_gt_box[3] / TARGET_IMAGE_SIZE[0]
                ]

                # Clamp predicted bounding box coordinates to [0, 1]
                normalized_pred_box = [
                    max(0, min(1, pred_bbox[0])),
                    max(0, min(1, pred_bbox[1])),
                    max(0, min(1, pred_bbox[2])),
                    max(0, min(1, pred_bbox[3]))
                ]

                print(f"Normalized Ground Truth BBox: {normalized_gt_box}")
                print(f"Normalized Predicted BBox: {normalized_pred_box}")

                # Visualize results
                visualize_results(image, normalized_gt_box, normalized_pred_box, TARGET_IMAGE_SIZE)

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

# --- Main ---
if __name__ == "__main__":
    # Define paths
    MODEL_PATH = "object_detection_model.h5"
    IMAGE_DIRECTORY = "../content/gdrive/MyDrive/MC_RCNN"
    ANNOTATION_FILE_PREFIX = "../content/gdrive/MyDrive/MC_RCNN/res_"

    # Load the label binarizer
    with open("lb.pickle", "rb") as f:
        lb = pickle.load(f)

    # Run inference
    run_inference(IMAGE_DIRECTORY, ANNOTATION_FILE_PREFIX, MODEL_PATH, lb)
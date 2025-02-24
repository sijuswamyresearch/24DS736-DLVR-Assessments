import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping

# --- Global Settings ---
TARGET_IMAGE_SIZE = (224, 224)  # Input size for MobileNetV2
IOU_THRESHOLD_POSITIVE = 0.70    # IoU threshold for positive samples
IOU_THRESHOLD_NEGATIVE = 0.3    # IoU threshold for negative samples
IMAGE_DIRECTORY = "../content/gdrive/MyDrive/MC_RCNN"
ANNOTATION_FILE = "../content/gdrive/MyDrive/MC_RCNN/res_"
CLASSES = ["butterfly", "dalmatian", "dolphin"]
MAX_IMAGES_PER_CLASS = 20       # Number of images to process per class
MAX_PROPOSALS = 20              # Max number of region proposals per image
BATCH_SIZE = 32                 # Batch size for training
NUM_EPOCHS = 20                 # Number of training epochs
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

# --- Process First 10 Images ---
def process_first_10_images(image_directory, annotation_file, classes, max_images_per_class, max_proposals):
    """
    Process the first 10 images from each class.
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    lb = LabelBinarizer()
    lb.fit(classes + ["background"])  # Include background class

    # Initialize lists to store data
    train_images = []
    train_labels = []
    train_bboxes = []

    # Iterate over each class
    for cl in classes:
        annotation_path = annotation_file + cl + ".txt"
        with open(annotation_path, "r") as f:
            lines = f.readlines()[:max_images_per_class]  # Process only the first 10 images
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

                    # Run Selective Search
                    ss.setBaseImage(resized_image)
                    ss.switchToSelectiveSearchFast()
                    ss_results = ss.process()[:max_proposals]  # Limit proposals

                    # Process Selective Search results
                    for result in ss_results:
                        x, y, w, h = result
                        proposal_box = [x, y, x + w, y + h]

                        # Calculate IoU
                        iou_score = get_iou({"x1": scaled_gt_box[0], "y1": scaled_gt_box[1], "x2": scaled_gt_box[2], "y2": scaled_gt_box[3]},
                                            {"x1": proposal_box[0], "y1": proposal_box[1], "x2": proposal_box[2], "y2": proposal_box[3]})

                        # Positive sample (IoU > 0.7)
                        if iou_score > IOU_THRESHOLD_POSITIVE:
                            roi = resized_image[y:y + h, x:x + w]
                            resized_roi = cv2.resize(roi, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                            train_images.append(resized_roi)
                            train_labels.append(cl)  # Use class label
                            train_bboxes.append(scaled_gt_box)  # Use scaled ground truth bounding box

                        # Negative sample (IoU < 0.3)
                        elif iou_score < IOU_THRESHOLD_NEGATIVE:
                            roi = resized_image[y:y + h, x:x + w]
                            resized_roi = cv2.resize(roi, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                            train_images.append(resized_roi)
                            train_labels.append("background")  # Use background label
                            train_bboxes.append([0, 0, 0, 0])  # Dummy bounding box for background

                except Exception as e:
                    print(f"Error processing {image_name}: {e}")
                    continue

    # Convert lists to numpy arrays
    train_images = np.array([preprocess_input(img_to_array(img)) for img in train_images])
    train_labels = lb.transform(train_labels)  # Encode labels (including background)
    train_bboxes = np.array(train_bboxes)

    # Normalize bounding box coordinates
    train_bboxes = train_bboxes / np.array([TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]])

    # Save the label binarizer
    with open("lb.pickle", "wb") as f:
        pickle.dump(lb, f)

    return train_images, train_labels, train_bboxes

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

# --- Plot Training History ---
def plot_training_history(history):
    """
    Plot training and validation loss/accuracy for both classification and bounding box regression.
    """
    plt.figure(figsize=(16, 12))

    # Classification Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history["class_label_loss"], label="Training Loss")
    plt.plot(history.history["val_class_label_loss"], label="Validation Loss")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Classification Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history["class_label_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_class_label_accuracy"], label="Validation Accuracy")
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Bounding Box Regression Loss
    plt.subplot(2, 2, 3)
    plt.plot(history.history["bounding_box_loss"], label="Training Loss")
    plt.plot(history.history["val_bounding_box_loss"], label="Validation Loss")
    plt.title("Bounding Box Regression Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Bounding Box Regression MAE
    plt.subplot(2, 2, 4)
    plt.plot(history.history["bounding_box_mae"], label="Training MAE")
    plt.plot(history.history["val_bounding_box_mae"], label="Validation MAE")
    plt.title("Bounding Box Regression MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()
# --- Main ---
if __name__ == "__main__":
    # Step 1: Process the first 10 images from each class
    print("[INFO] Processing the first 10 images from each class...")
    train_images, train_labels, train_bboxes = process_first_10_images(
        IMAGE_DIRECTORY, ANNOTATION_FILE, CLASSES, MAX_IMAGES_PER_CLASS, MAX_PROPOSALS
    )
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Training bounding boxes shape: {train_bboxes.shape}")

    # Step 2: Split data into training and validation sets
    print("[INFO] Splitting data...")
    (
        trainX,
        valX,
        trainY,
        valY,
        trainBbox,
        valBbox,
    ) = train_test_split(
        train_images, train_labels, train_bboxes, test_size=0.2, random_state=42
    )

    # Step 3: Build the model
    print("[INFO] Building model...")
    model = build_model(TARGET_IMAGE_SIZE + (3,), len(CLASSES) + 1)  # +1 for background class
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        optimizer=opt,
        loss={
            "class_label": "categorical_crossentropy",
            "bounding_box": "mse",
        },
        loss_weights={
            "class_label": 1.0,
            "bounding_box": 10.0,  # Increased weight for bounding box loss
        },
        metrics={
            "class_label": "accuracy",
            "bounding_box": "mae",  # Mean Absolute Error for bounding box regression
        },
    )

    # Step 4: Train the model
    print("[INFO] Training model...")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        trainX,
        {"class_label": trainY, "bounding_box": trainBbox},
        validation_data=(valX, {"class_label": valY, "bounding_box": valBbox}),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Step 5: Plot training history
    print("[INFO] Plotting training history...")
    plot_training_history(history)

    # Step 6: Save the model
    print("[INFO] Saving model...")
    model.save("object_detection_model.h5")
    print("[INFO] Model saved.")
import imutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2  # Using MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNetV2 preprocessing

# --- Global Settings ---
TARGET_IMAGE_SIZE = (224, 224)  # MobileNetV2 expects (224, 224, 3)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
IMAGE_DIRECTORY = "../content/gdrive/MyDrive/MC_RCNN"
NUM_EPOCHS = 25
INIT_LR = 1e-4
CLASSES = ["butterfly", "dalmatian", "dolphin"]

# --- Load Dataset ---
data = []
labels = []
bboxes = []
imagePaths = []

ann_path = "../content/gdrive/MyDrive/MC_RCNN/res_"
images_path = "../content/gdrive/MyDrive/MC_RCNN"

for cl in CLASSES:
    ann_path_new = ann_path + cl + ".txt"
    print(ann_path_new)
    rows = open(ann_path_new).read().strip().split("\n")

    # Loop over the rows
    for idx, row in enumerate(rows):
        # Break the row into the filename and bounding box coordinates
        row = row.split(",")
        filename = row[0]
        filename = filename.split(".")[0]
        filename = filename.split("_")[-1]
        filename = "image_" + filename + ".jpg"

        coords = row[1:]
        coords = [int(c) for c in coords]

        label = cl
        image_path = os.path.sep.join([images_path, cl, filename])

        # Load and preprocess the image
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]

        # Scale the bounding box coordinates relative to the image dimensions
        Xmin = float(coords[0]) / w
        Ymin = float(coords[1]) / h
        Xmax = float(coords[2]) / w
        Ymax = float(coords[3]) / h

        # Resize the image to the target size
        image = load_img(image_path, target_size=TARGET_IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)  # MobileNetV2 preprocessing

        # Append to the dataset
        data.append(image)
        labels.append(label)
        bboxes.append((Xmin, Ymin, Xmax, Ymax))
        imagePaths.append(image_path)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# One-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
if len(lb.classes_) == 2:  # Handle binary classification
    labels = to_categorical(labels)

# --- Split Data ---
(trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes, trainPaths, testPaths) = train_test_split(
    data, labels, bboxes, imagePaths, test_size=0.20, random_state=42)

# --- Model Building ---
# Define the input tensor explicitly
input_tensor = Input(shape=INPUT_SHAPE, name="input_tensor")

# Load MobileNetV2 with the defined input tensor
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
baseModel.trainable = False  # Freeze the base model

# Add custom head for classification and bounding box regression
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

# Classification head
classHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(headModel)

# Bounding box regression head
bboxHead = Dense(128, activation="relu")(headModel)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# Define the model
model = Model(inputs=input_tensor, outputs=[classHead, bboxHead])

# Print model summary
model.summary()

# --- Compile the Model ---
opt = Adam(INIT_LR)
model.compile(optimizer=opt,
              loss={"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"},
              metrics={"class_label": "accuracy", "bounding_box": "mae"},
              loss_weights={"class_label": 1.0, "bounding_box": 1.0})

# --- Train the Model ---
history = model.fit(
    trainImages, {"class_label": trainLabels, "bounding_box": trainBBoxes},
    validation_data=(testImages, {"class_label": testLabels, "bounding_box": testBBoxes}),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1)

# --- Save the Model ---
model.save("model_bbox_regression_and_classification_mobi.h5")
f = open("lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

# --- Test the Model ---
# Load the model and label binarizer
model = load_model("model_bbox_regression_and_classification_mobi.h5")
lb = pickle.loads(open("lb.pickle", "rb").read())

# Load test image paths
testPaths = open("testing_multiclass.txt").read().strip().split("\n")

for imagePath in testPaths:
    # Load and preprocess the test image
    original_image = cv2.imread(imagePath)
    (orig_h, orig_w) = original_image.shape[:2]

    image = load_img(imagePath, target_size=TARGET_IMAGE_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)  # MobileNetV2 preprocessing
    image = np.expand_dims(image, axis=0)

    # Predict bounding box and class
    (labelPreds, boxPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # Scale the bounding box coordinates to the original image size
    startX = int(startX * orig_w)
    startY = int(startY * orig_h)
    endX = int(endX * orig_w)
    endY = int(endY * orig_h)

    # Determine the class label
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # Draw the bounding box and label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(original_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.show()
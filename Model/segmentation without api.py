import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define the path to your dataset folder
dataset_path = "/path/to/your/dataset/folder"

# Define the confidence threshold for labeling (90%)
confidence_threshold = 0.9

# Load your trained CNN model (modify the model loading code accordingly)
model = load_model("terrain_classification_model.h5")

# Define a mapping of class IDs to labels
class_labels = {
    0: "Sandy",
    1: "Rocky",
    2: "Grassy",
    3: "Marshy"
}

# Function to classify and label sub-segments
def classify_and_label(image):
    # Resize and preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    
    # Make a prediction using the model
    prediction = model.predict(np.expand_dims(image, axis=0))
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Get the class label based on the predicted class ID
    class_label = class_labels[class_id]
    
    return class_label, confidence

# Recursive function to divide and label sub-segments
def divide_and_label(image, x, y, width, height):
    sub_image = image[y:y+height, x:x+width]
    class_label, confidence = classify_and_label(sub_image)

    # If confidence is below the threshold or the image is too small, stop subdividing
    if confidence < confidence_threshold or min(width, height) <= 16:
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display percentage recognition for each class
        class_probabilities = model.predict(np.expand_dims(sub_image, axis=0))[0]
        for class_id, probability in enumerate(class_probabilities):
            class_name = class_labels[class_id]
            text = f"{class_name}: {probability*100:.2f}%"
            cv2.putText(image, text, (x, y + 15 + class_id * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Divide the sub-segment into four parts
        new_width = width // 2
        new_height = height // 2
        divide_and_label(image, x, y, new_width, new_height)
        divide_and_label(image, x+new_width, y, new_width, new_height)
        divide_and_label(image, x, y+new_height, new_width, new_height)
        divide_and_label(image, x+new_width, y+new_height, new_width, new_height)

# Load and process an image from the dataset folder
image_file = "your_image.jpg"  # Replace with your image file name
image_path = os.path.join(dataset_path, image_file)

if os.path.isfile(image_path):
    # Load the image
    segmented_image = cv2.imread(image_path)

    # Initialize the image labeling process
    divide_and_label(segmented_image, 0, 0, segmented_image.shape[1], segmented_image.shape[0])

    # Save the segmented and labeled image
    cv2.imwrite("segmented_and_labeled_image.jpg", segmented_image)
else:
    print(f"Image not found at path: {image_path}")

# Display the labeled image using matplotlib
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

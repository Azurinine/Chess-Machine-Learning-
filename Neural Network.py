import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from keras.regularizers import l2
from keras.layers import Input, Concatenate
from keras.models import Model

# Load the processed data and labels
image_array = np.load('processed_data.npy')
labels = np.load('processed_labels.npy')

# Apply edge detection to each image using the Canny edge detector
def apply_edge_detection(image):
    # Convert the image to 8-bit unsigned integer data type
    image = image.astype(np.uint8)
    
    # Apply Canny edge detection
    return cv2.Canny(image, 100, 200)  # You can adjust the threshold values

# Apply edge detection to all images
edge_image_array = [apply_edge_detection(image) for image in image_array]

# Convert the list of edge images into a numpy array
edge_image_array = np.array(edge_image_array)

# Optionally, normalize the pixel values to a range of [0, 1]
edge_image_array = edge_image_array / 255.0

# Flatten the edge detection images
edge_image_array_flat = edge_image_array.reshape(-1, 45*45)

# Reshape the edge images
X_edge = np.array(edge_image_array).reshape(-1, 45, 45, 1)

# Create a mapping from class labels to integers
class_to_int = {label: i for i, label in enumerate(set(labels))}
int_to_class = {i: label for label, i in class_to_int.items()}

# Convert class labels to integers
y = np.array([class_to_int[label] for label in labels])

# Calculate the number of unique classes
num_classes = len(set(labels))

# One-hot encode the labels
y = to_categorical(y, num_classes)

# Reshape the images to match the input size (e.g., 45x45 pixels with 1 channel for grayscale)
X = image_array.reshape(-1, 45, 45, 1)

# Split the data into training and test sets
X_train, X_test, X_edge_train, X_edge_test, y_train, y_test = train_test_split(X, X_edge, y, test_size=0.2, random_state=42)

# Define the model architecture using the Functional API
input_grayscale = Input(shape=(45, 45, 1), name='input_grayscale')
input_edge = Input(shape=(45, 45, 1), name='input_edge')

# Branch for grayscale images
x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_grayscale)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(64, activation='relu')(x1)

# Branch for edge detection images
x2 = layers.Conv2D(32, (3, 3), activation='relu')(input_edge)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(64, activation='relu')(x2)

# Concatenate both branches
merged = Concatenate()([x1, x2])

# Continue with the combined model
merged = layers.Dense(200, activation='sigmoid')(merged)
merged = layers.Dense(200, activation='relu')(merged)

output = layers.Dense(num_classes, activation='softmax')(merged)

model = Model(inputs=[input_grayscale, input_edge], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit({'input_grayscale': X_train, 'input_edge': X_edge_train}, y_train, epochs=50, validation_split=0.3)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate({'input_grayscale': X_test, 'input_edge': X_edge_test}, y_test)

print(f"Test accuracy: {test_accuracy:.2f}")

# Predict the labels for the test set
y_pred = model.predict({'input_grayscale': X_test, 'input_edge': X_edge_test})
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Convert integer labels back to class labels
y_pred_labels = [int_to_class[i] for i in y_pred_classes]
y_test_labels = [int_to_class[i] for i in y_test_classes]

# Create a confusion matrix and classification report
confusion = confusion_matrix(y_test_labels, y_pred_labels)
classification_rep = classification_report(y_test_labels, y_pred_labels)

# Display the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

# Save the predictions and actual labels to a CSV file
df = pd.DataFrame({'Actual': y_test_labels, 'Predicted': y_pred_labels})
df.to_csv('predictions.csv', index=False)

# Identify the incorrect predictions
incorrect_indices = [i for i in range(len(y_test_labels)) if y_test_labels[i] != y_pred_labels[i]]

# Display some of the incorrect predictions
num_samples_to_display = min(5, len(incorrect_indices))
for i in range(num_samples_to_display):
    index = incorrect_indices[i]
    plt.imshow(X_test[index].reshape(45, 45), cmap='gray')
    plt.title(f"Actual: {y_test_labels[index]}, Predicted: {y_pred_labels[index]}")
    plt.show()
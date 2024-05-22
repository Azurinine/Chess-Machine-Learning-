import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

root_data_dir = '/Users/achintya/Downloads/png_piece_achintya/'  # Replace this with the actual path to the chess pieces in the files

# Initialize empty lists to store labels and image data
labels = []
image_array = []

def augment_image(image, width=45, height=45):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the specified width and height
    resized_image = cv2.resize(grayscale_image, (width, height))
    
    # Apply distortion (you can adjust the parameters)
    rows, cols = resized_image.shape
    k = random.uniform(0.5, 1)  # Control the intensity of distortion
    dx = k * np.random.randn()
    dy = k * np.random.randn()
    distorted_image = cv2.warpAffine(resized_image, np.float32([[1, 0, dx], [0, 1, dy]]), (cols, rows))
    
    # Apply horizontal flip
    flipped_horizontal = cv2.flip(resized_image, 1)
    
    # Apply vertical flip
    flipped_vertical = cv2.flip(resized_image, 0)

    return [resized_image, flipped_horizontal, flipped_vertical, distorted_image]

def display_images(images, labels=None, ncols=4, figsize=(10, 6), background_color='white'):
    num_images = len(images)
    nrows = (num_images + ncols - 1) // ncols  # Calculate the number of rows

    plt.figure(figsize=figsize)
    plt.gca().set_facecolor(background_color)  # Set the background color

    for i in range(num_images):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i], cmap='gray')  # Use cmap='gray' for grayscale images
        plt.axis('off')  # Turn off axis labels
        if labels is not None:
            plt.title(labels[i])

    plt.tight_layout()
    plt.show()

# Traverse through all subdirectories
for root, dirs, files in os.walk(root_data_dir):
    for filename in files:
        if filename.endswith((".jpg", ".png")):  # Make sure it's a JPEG or PNG file
            image_path = os.path.join(root, filename)

            # Extract the label from the file name (e.g., "bB.png" -> "bB")
            label = os.path.splitext(filename)[0]

            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Apply augmentation (grayscale conversion, resize, distortion, horizontal and vertical flips)
            augmented_images = augment_image(image, width=45, height=45)

            # Append augmented images to the respective lists
            labels.extend([label] * 4)
            image_array.extend(augmented_images)

# Convert the lists of images and labels into numpy arrays
image_array = np.array(image_array)

# Optionally, normalize the pixel values to a range of [0, 1]
image_array = image_array / 255.0

# Display the augmented images
#display_images(image_array[:12], labels[:12], figsize=(8, 4))

print(image_array.shape)

# Save the processed data (image_array and labels) to a file
np.save('processed_data.npy', image_array)
np.save('processed_labels.npy', labels)

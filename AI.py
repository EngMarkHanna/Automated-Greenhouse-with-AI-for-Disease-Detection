import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

NUM_OF_CLASS = 0 #Need to fill based on AI


# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values
    rotation_range=15,  # Random rotations
    width_shift_range=0.1,  # Horizontal shifts
    height_shift_range=0.1,  # Vertical shifts
    shear_range=0.2,  # Shearing
    zoom_range=0.2,  # Zooming
    horizontal_flip=True,  # Horizontal flips
    fill_mode='nearest'  # Fill mode for transformations
)

# Simple normalization for testing set
val_datagen = ImageDataGenerator(
    rescale=1.0/255  # Normalize pixel values
)

# Load the data for training and testing
train_generator = train_datagen.flow_from_directory(
    directory= "Fulldata/train/Cucumber",
    target_size=(299, 299),  # InceptionV3 expects 299x299 images
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=True  # Shuffle the data
)

val_generator = val_datagen.flow_from_directory(
    directory = "Fulldata/val/Cucumber",
    target_size=(299, 299),  # Ensure consistent image size
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    shuffle=False  # Do not shuffle validation data
)

# Load the InceptionV3 model with pre-trained weights and exclude the top layers
base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',  # Load pre-trained weights from ImageNet
    include_top=False,  # Exclude the top layers (fully connected layers)
    input_shape=(299, 299, 3)  # Input shape for InceptionV3
)

# Set the base model as non-trainable (freeze layers)
base_model.trainable = False

# Add custom layers to the model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)  # Pooling to reduce dimensionality
x = layers.Dropout(0.5)(x)  # Dropout for regularization
x = layers.Dense(256, activation='relu')(x)  # Fully connected layer
x = layers.Dropout(0.5)(x)  # Additional Dropout
output_layer = layers.Dense(NUM_OF_CLASS, activation='softmax')(x)  # Output layer (adjust number of classes)

# Create the final model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(
    optimizer='adam',  # Choose an optimizer
    loss='categorical_crossentropy',  # Appropriate loss for multi-class classification
    metrics=['accuracy']  # Track accuracy
)

# Train the model with the data generators
model.fit(
    train_generator,
    #validation_data=val_generator,
    epochs=10  # Adjust the number of epochs as needed
)

model.summary()

import numpy as np

# Predict on the testing set
predictions = model.predict(val_generator)

# Since we're using categorical labels, we need to convert predictions to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get the true labels from the testing set
true_classes = val_generator.classes  # This gives you the actual class indices

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))  # Adjust size as needed
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.title('Confusion Matrix')
plt.show()

model.save('name.h5') 
#!/usr/bin/env python
# coding: utf-8

# ## Handwritten Digit Recognizer

# ## Author: Syed Abbas Ali

# ### Import necessary libraries

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ### Load and Prepare the Data

# In[18]:


# Load the training dataset
train_data = pd.read_csv(r"C:\Users\HP\Downloads\digit-recognizer\train.csv")
train_data.head(5)


# In[19]:


train_data.shape


# In[20]:


test_data=pd.read_csv(r"C:\Users\HP\Downloads\digit-recognizer\test.csv")
test_data.head(5)


# In[21]:


test_data.shape


# In[22]:


# Separate the "label" column from the training dataset
y_train = train_data['label']

# Split the remaining columns into features (X_train)
X_train = train_data.drop(columns=['label'])

# Normalize the pixel values to bring them within a common scale (between 0 and 1)
X_train = X_train / 255.0

# Reshape the data for CNN
X_train = X_train.values.reshape(-1, 28, 28, 1)


# ### Data Visualization

# In[25]:


# Display some sample training images along with their labels
def plot_sample_images(x, y, num_imgs=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_imgs):
        img = x[i].reshape(28, 28)
        plt.subplot(1, num_imgs, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y.iloc[i]}")
        plt.axis("off")
    plt.show()

plot_sample_images(X_train, y_train, num_imgs=5)


# ### Model Selection and Training

# In[27]:


# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))

# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Calculate evaluation metrics on the validation set
val_loss, val_accuracy = cnn_model.evaluate(X_val, y_val)


# ### Model Evaluation

# In[34]:


# Print evaluation metrics
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Predict on the validation set
y_val_pred = cnn_model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Calculate evaluation metrics
val_precision = precision_score(y_val, y_val_pred_classes, average='weighted')
val_recall = recall_score(y_val, y_val_pred_classes, average='weighted')
val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')

# Print evaluation metrics
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1-score:", val_f1)

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_classes))

# Generate a confusion matrix
confusionMatrix = confusion_matrix(y_val, y_val_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusionMatrix, annot=True, linewidths=0.1, cmap="cividis", linecolor="black", fmt='.0f')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# ### Predictions on Test Data with Visualization

# In[29]:


# Normalize the pixel values of test_data
X_test = test_data / 255.0

# Reshape the test data to a 28x28x1 image format
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Predict the labels for the test dataset (X_test)
y_test_pred = cnn_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Visualize some sample test images along with their predicted labels
fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    img = X_test[i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set(title=f"Predicted Number is: {y_test_pred_classes[i]}")
plt.show()


# In[31]:


predictions_df = pd.DataFrame({'ImageId': np.arange(1, len(X_test) + 1), 'PredictedLabel': y_test_pred_classes})
predictions_df.to_csv("predictions_hdr.csv", index=False)


# In[32]:


hdr=pd.read_csv(r"C:\Users\HP\Downloads\predictions_hdr.csv")
hdr.head(5)


# In[ ]:





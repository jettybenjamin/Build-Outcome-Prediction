import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Load the 'cloudify.csv' dataset
data = pd.read_csv('openproject.csv')

# Convert datetime column to datetime type
data['gh_build_started_at'] = pd.to_datetime(data['gh_build_started_at'])

# Extract features and target
features = data.drop('build_Failed', axis=1)  # Features: all columns except 'build_Failed'
target = data['build_Failed']  # Target column

# Drop non-numeric or non-relevant columns (like datetime) from features
features = features.drop('gh_build_started_at', axis=1)  # Drop datetime column

# Convert features and target to numpy arrays
features_array = features.to_numpy()  # Convert features to numpy array
target_array = target.to_numpy()  # Convert target to numpy array

# Ensure all data types are compatible with TensorFlow
features_array = features_array.astype('float32')  # Convert features to float32 if needed
target_array = target_array.astype('int32')  # Convert target to int32 if needed

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_array, target_array, test_size=0.2, random_state=42)

# Create the LSTM-GRU hybrid model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(GRU(50))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
from sklearn.metrics import classification_report

# Make predictions on the validation set
y_pred_prob = model.predict(X_val)

# Threshold probabilities to get binary predictions (0 or 1)
threshold = 0.5  # Adjust this threshold based on your problem
y_pred = (y_pred_prob > threshold).astype('int32')

# Generate classification report
report = classification_report(y_val, y_pred)
print(report)

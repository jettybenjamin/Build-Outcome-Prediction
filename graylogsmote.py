import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('graylog2-server.csv')

# Convert datetime column to datetime type
data['gh_build_started_at'] = pd.to_datetime(data['gh_build_started_at'])

# Extract features and target
features = data.drop(['build_Failed', 'gh_build_started_at'], axis=1)  # Drop datetime column and target
target = data['build_Failed']  # Target column

# Convert features and target to numpy arrays
features_array = features.to_numpy().astype('float32')  # Convert features to float32
target_array = target.to_numpy().astype('int32')  # Convert target to int32

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_array, target_array, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_sm), y=y_train_sm)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Reshape input data for LSTM
X_train_sm = X_train_sm.reshape(X_train_sm.shape[0], X_train_sm.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Create the LSTM-GRU hybrid model with dropout for regularization
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train_sm.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with class weights and early stopping
model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=32, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])

# Make predictions on the validation set
y_pred_prob = model.predict(X_val)

# Threshold probabilities to get binary predictions (0 or 1)
threshold = 0.5  # Adjust this threshold based on your problem
y_pred = (y_pred_prob > threshold).astype('int32')

# Generate classification report
report = classification_report(y_val, y_pred)
print(report)

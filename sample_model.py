import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load and preprocess the dataset
file_path = r'area_cum.xlsx'
data = pd.read_excel(file_path)

data.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for col in data.columns[1:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna().reset_index(drop=True)

# Features and target
X = data.iloc[:, 1:]  # Monthly data
y = X.mean(axis=1)    # Average groundwater level

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM input
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Build the LSTM model
model = Sequential([
    Bidirectional(LSTM(128, activation='tanh', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1))),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks for better training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=200, 
    batch_size=16, 
    validation_split=0.2, 
    verbose=1, 
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Make predictions
predictions = model.predict(X_test_scaled)

# Rescale predictions and actual values back to original scale
predictions_rescaled = predictions.flatten()
actuals_rescaled = y_test.values

# Display predictions and actuals
for i in range(len(predictions_rescaled)):
    print(f"Predicted: {predictions_rescaled[i]:.2f}, Actual: {actuals_rescaled[i]:.2f}")

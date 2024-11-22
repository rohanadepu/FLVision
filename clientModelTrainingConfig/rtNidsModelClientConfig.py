import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.regularizers import l2

# Hyperparameters
l2_alpha = 0.01  # L2 regularization factor
input_dim = 22  # Number of features
time_steps = 50  # Length of time-series window
dropout_rate = 0.4

# Build the enhanced model
model = tf.keras.Sequential([
    # Input layer for time-series data
    tf.keras.layers.Input(shape=(time_steps, input_dim)),

    # 1D Convolutional Layer for feature extraction
    Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # Another Conv1D Layer
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # Recurrent Layer (LSTM)
    LSTM(100, return_sequences=False, activation='tanh', kernel_regularizer=l2(l2_alpha)),
    Dropout(dropout_rate),

    # Fully connected layers
    Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    Dropout(dropout_rate),

    Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    Dropout(dropout_rate),

    Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    Dropout(dropout_rate),

    # Output layer for binary classification
    Dense(1, activation='sigmoid')
])

# Build the real-time sequential model
model = Sequential([
    # Input layer for real-time time-series data
    tf.keras.layers.Input(shape=(time_steps, input_dim)),

    # 1D Convolutional Layer for feature extraction
    Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # Another Conv1D Layer
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    # GRU Layer (lighter than LSTM for real-time application)
    GRU(50, return_sequences=False, activation='tanh', kernel_regularizer=l2(l2_alpha)),
    Dropout(dropout_rate),

    # Fully connected layers for classification
    Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    Dropout(dropout_rate),

    Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
    BatchNormalization(),
    Dropout(dropout_rate),

    # Output layer for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

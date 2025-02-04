import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.regularizers import l2

# Hyperparameters
l2_alpha = 0.01  # L2 regularization factor
input_dim = 22  # Number of features
time_steps = 50  # Length of time-series window
dropout_rate = 0.4


def create_auto_feature_extraction_realtime_LSTM_Model(input_dim, l2_alpha, dropout_rate, time_steps):
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

    return model


def create_auto_feature_extraction_realtime_GRU_Model(input_dim, l2_alpha, dropout_rate, time_steps):
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

    return model

def create_realtime_GRU_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha):

    # with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),  # Dropout layer with 50% dropout rate

        Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    return model


# ---                   CICIOT Models                   --- #
def create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha):

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),  # Dropout layer with 50% dropout rate

            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(1, activation='sigmoid')
        ])

    elif regularizationEnabled and DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

    elif DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    return model


# ---                   IOTBOTNET Models                  --- #

def create_IOTBOTNET_Model(input_dim, regularizationEnabled, l2_alpha):

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 30% dropout rate
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 50% dropout rate
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    return model
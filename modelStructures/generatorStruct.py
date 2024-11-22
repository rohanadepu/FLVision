# Function for creating the generator model
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator


def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(noise_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),

        # Adding an LSTM layer to better generate realistic time-series data
        tf.keras.layers.Reshape((1, 512)),  # Reshape to use LSTM after fully connected layers
        LSTM(50, return_sequences=False, activation='tanh'),

        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator
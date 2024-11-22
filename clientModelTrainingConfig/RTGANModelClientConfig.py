def create_discriminator(input_dim):
    # Discriminator to classify three classes: Normal, Intrusive, Fake
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer for three classes
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])
    return discriminator


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

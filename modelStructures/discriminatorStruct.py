# Function for creating the discriminator model
def create_discriminator(input_dim):
    # Discriminator is designed to classify three classes:
    # - Normal (Benign) traffic
    # - Intrusive (Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])
    return discriminator

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
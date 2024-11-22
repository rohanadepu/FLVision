def create_model(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator(input_dim))

    return model


def load_GAN_model(generator, discriminator):
    model = Sequential([generator, discriminator])

    return model


def split_GAN_model(model):
    # Assuming `self.model` is the GAN model created with Sequential([generator, discriminator])
    generator = model.layers[0]
    discriminator = model.layers[1]

    return generator, discriminator
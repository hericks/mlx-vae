import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist

class MLPEncoder(nn.Module):
    """A simple MLP encoder."""

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int
    ):
        super().__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def __call__(self, x):
        x = self.hidden(x)
        x = mx.maximum(x, 0.0)
        return self.mean(x), self.logvar(x)
    

class MLPDecoder(nn.Module):
    """A simple MLP decoder."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def __call__(self, z):
        z = self.hidden(z)
        z = mx.maximum(z, 0.0)
        z = self.output(z)
        return mx.sigmoid(z)
    

class VAE(nn.Module):
    """A simple VAE with MLP encoder and decoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim)

    def __call__(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Sample from the latent space."""
        return mean + mx.exp(0.5 * logvar) * mx.random.normal(mean.shape)


if __name__ == "__main__":
    hidden_dim = 32
    latent_dim = 6

    # Load the data
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())    

    # Load the model
    vae = VAE(train_images.shape[-1], hidden_dim, latent_dim)
    mx.eval(vae.parameters())
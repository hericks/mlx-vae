import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np
import matplotlib.pyplot as plt

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


def loss_fn(vae, X):
    recon, mean, logvar = vae(X)
    recon_loss = 0.5 * mx.sum(mx.square(X - recon), axis=-1)
    kl_loss = - 0.5 * mx.sum(1 + logvar - mean ** 2 - mx.exp(logvar), axis=-1)
    return mx.mean(recon_loss + kl_loss)


def sample_fn(vae):
    zs = mx.random.normal((5, 6))
    samples = vae.decoder(zs)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for ax, s in zip(axs, samples):
        ax.imshow(s.reshape(28, 28))

    fig.savefig("vae.png")
    plt.close(fig)


def batch_iterate(batch_size, X):
    perm = mx.array(np.random.permutation(X.shape[0]))
    for s in range(0, X.shape[0], batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids]


if __name__ == "__main__":
    hidden_dim = 128
    latent_dim = 6

    num_epochs = 100
    batch_size = 128

    learning_rate = 3e-4

    mx.set_default_device(mx.cpu)

    # Load the data
    train_images, _, test_images, _ = map(mx.array, mnist.mnist())    

    # Load the model
    vae = VAE(train_images.shape[-1], hidden_dim, latent_dim)
    mx.eval(vae.parameters())

    vae_loss_and_grad_fn = nn.value_and_grad(vae, loss_fn)
    optimizer = optim.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        tic = time.perf_counter()
        for i, X in enumerate(batch_iterate(batch_size, train_images)):
            loss, grads = vae_loss_and_grad_fn(vae, X)
            optimizer.update(vae, grads)
            mx.eval(vae.parameters(), optimizer.state)
        sample_fn(vae)
        toc = time.perf_counter()
        print(
            f"Epoch {epoch}:",
            f" Loss {loss},",
            f" Time {toc - tic:.3f} (s)"
        )
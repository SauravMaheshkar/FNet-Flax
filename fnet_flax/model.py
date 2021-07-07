import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from jax import lax

__all__ = ["FeedForward", "EncoderBlock", "FNet"]


class FeedForward(nn.Module):

    dim: int
    expansion_factor: int
    dropout_rate: jnp.float32

    def setup(self):
        self.fc1 = nn.Dense(features=self.expansion_factor * self.dim)
        self.fc2 = nn.Dense(features=self.dim)
        self.drop = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, deterministic=False) -> Array:

        out = self.fc1(x)
        out = nn.gelu(out)
        out = self.fc2(out)
        output = self.drop(out, deterministic=deterministic)

        return output


class EncoderBlock(nn.Module):

    d_hidden: int = 512

    def setup(self):
        self.ff = FeedForward(dim=self.d_hidden, expansion_factor=4, dropout_rate=0.1)

    @nn.compact
    def __call__(self, x):

        x_fft = lax.real(jnp.fft.fft2(x, axes=(-1, -2)))
        x = nn.LayerNorm(name="LN1")(x + x_fft)
        x_ff = self.ff(x)
        x = nn.LayerNorm(name="LN2")(x + x_ff)
        return x


class FNet(nn.Module):

    depth: int
    dim: int

    def setup(self):

        self.layers = [EncoderBlock(d_hidden=self.dim) for _ in range(self.depth)]
        self.dense = nn.Dense(features=self.dim)

    @nn.compact
    def __call__(self, x) -> Array:

        for layer in self.layers:
            x = layer(x)

        output = self.dense(x)
        return output

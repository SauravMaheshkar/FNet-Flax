import flax.linen as nn
import numpy as np
from jax import random

from fnet_flax import FNet


def test_instance():

    model = FNet(depth=2, dim=32)
    assert isinstance(model, nn.Module)


def test_init():

    x = np.random.randn(2, 8, 32)
    init_rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
    model = FNet(depth=2, dim=32).init(init_rngs, x)  # noqa: F841

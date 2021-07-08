![](https://github.com/SauravMaheshkar/FNet-Flax/blob/main/assets/fnet.jpeg?raw=true)

In this [paper](https://arxiv.org/abs/2105.03824), the authors introduce FNet, a new architecture inspired by Transformers where the learnable Self-Attention layer has been replaced by a unlearnable Fourier Transform. This work highlights the potential of linear units as a drop-in replacement for the attention mechanism. They particularly found Fourier Transforms to be an effective mixing mechanism, in part due to the highly efficient FFT. Remarkably, this unparameterized mixing mechanism can yield relatively competitive models. This work among many is particularly important for deploying transformer like models to low-compute scenarios.

## Installation

You can install this package from PyPI:

```sh
pip install fnet-flax
```

Or directly from GitHub:

```sh
pip install --upgrade git+hhttps://github.com/SauravMaheshkar/FNet-Flax.git
```

## Usage

```python
import numpy as np
from jax import random
from fnet_flax import FNet

x = np.random.randn(2, 8, 32)
init_rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
model = FNet(depth=2, dim=32).init(init_rngs, x)
```

## Development

### 1. Conda Approach

```sh
conda env create --name <env-name> sauravmaheshkar/fnet
conda activate <env-name>
```

### 2. Docker Approach

```sh
docker pull ghcr.io/sauravmaheshkar/fnet-dev:latest
docker run -it -d --name <container_name> ghcr.io/sauravmaheshkar/fnet-dev
```

Use the [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension in VSCode and [attach to the running container](https://code.visualstudio.com/docs/remote/attach-container). The code resides in the `code/` dir.

Alternatively you can also download the image from [Docker Hub](https://hub.docker.com/r/sauravmaheshkar/fnet-dev).

```sh
docker pull sauravmaheshkar/fnet-dev
```

## Citations

```bibtex
@misc{leethorp2021fnet,
      title={FNet: Mixing Tokens with Fourier Transforms},
      author={James Lee-Thorp and Joshua Ainslie and Ilya Eckstein and Santiago Ontanon},
      year={2021},
      eprint={2105.03824},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

import jax
import jax.numpy as jnp
from flax import nnx

from config import get_config


class ChessCNN(nnx.Module):
    def __init__(self, config, in_planes: int, rngs: nnx.Rngs):
        kernel = (config.kernel_size, config.kernel_size)

        convs = [
            nnx.Conv(
                in_features=in_planes,
                out_features=config.features,
                kernel_size=kernel,
                strides=config.stride,
                padding=config.padding,
                rngs=rngs,
            )
        ]

        for _ in range(config.num_layers - 1):
            convs.append(
                nnx.Conv(
                    in_features=config.features,
                    out_features=config.features,
                    kernel_size=kernel,
                    strides=config.stride,
                    padding=config.padding,
                    rngs=rngs,
                )
            )

        self.convs = nnx.List(convs)


if __name__ == "__main__":
    pass

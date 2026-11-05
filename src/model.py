import jax
import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import to_onnx

from checkpoint import PROJECT_ROOT, make_manager, restore_checkpoint
from config import get_config
from selfplay.train import make_optimizer


class ChessCNN(nnx.Module):
    def __init__(self, config, in_planes: int, num_moves: int, rngs: nnx.Rngs):
        kernel = (config.kernel_size, config.kernel_size)

        #Initalise the first convolution layer
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

        #Initalise hidden convolution layers
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

        #The size of conv output feature map, flattened before the heads
        flat_features = 8 * 8 * config.features

        #Initialise the policy head: hidden layers (could be 0), then one score per move
        policy_layers = []
        in_features = flat_features
        for _ in range(config.policy_head.hidden.num):
            policy_layers.append(
                nnx.Linear(
                    in_features=in_features,
                    out_features=config.policy_head.hidden.size,
                    rngs=rngs,
                )
            )
            in_features = config.policy_head.hidden.size  # next layer takes this layer's output

        #Final layer to compress into the amount of legal moves
        policy_layers.append(
            nnx.Linear(
                in_features=in_features,
                out_features=num_moves,
                rngs=rngs,
            )
        )

        #Initialise the value head: hidden layers (could be 0), then one eval number
        value_layers = []
        in_features = flat_features
        for _ in range(config.value_head.hidden.num):
            value_layers.append(
                nnx.Linear(
                    in_features=in_features,
                    out_features=config.value_head.hidden.size,
                    rngs=rngs,
                )
            )
            in_features = config.value_head.hidden.size  # next layer takes this layer's output

        #Final layer to compress into 1 evaluation of the chess board
        value_layers.append(
            nnx.Linear(
                in_features=in_features,
                out_features=1,
                rngs=rngs,
            )
        )

        self.convs = nnx.List(convs)
        self.policy_head = nnx.List(policy_layers)
        self.value_head = nnx.List(value_layers)

    def __call__(self, x):
        for conv in self.convs:
            x = nnx.relu(conv(x))

        #Flatten the feature maps for the heads: (batch, 8, 8, features) -> (batch, 8*8*features)
        x = x.reshape(x.shape[0], -1)

        #Policy: relu between layers, none on the last so move scores stay raw (softmax comes later)
        policy = x
        for i, layer in enumerate(self.policy_head):
            policy = layer(policy)
            if i < len(self.policy_head) - 1:
                policy = nnx.relu(policy)

        #Value: relu between layers, tanh on the last to keep the eval in [-1, 1]
        value = x
        for i, layer in enumerate(self.value_head):
            value = layer(value)
            if i < len(self.value_head) - 1:
                value = nnx.relu(value)
        value = jnp.tanh(value)

        return policy, value


#Writes the model with its current weights to one .onnx file (weights included, no .data sidecar)
#Input "board": float32 (batch, 8, 8, in_planes), "B" = any batch size; outputs "policy" and "value"
def export_onnx(model, path, in_planes: int):
    proto = to_onnx(model, [("B", 8, 8, in_planes)], input_names=["board"], output_names=["policy", "value"])
    onnx.save_model(proto, str(path), save_as_external_data=False)  # jax2onnx would split weights >1MB into .data
    return path


if __name__ == "__main__":
    #Exports the latest checkpoint to chess.onnx in the project root (random weights if none saved yet)
    config = get_config()
    model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
    restore_checkpoint(make_manager(config.training), model, make_optimizer(model, config.training))  # optimizer needed to load, then unused
    print("saved", export_onnx(model, PROJECT_ROOT / "chess.onnx", config.data.in_planes))

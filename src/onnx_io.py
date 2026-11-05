import sys

import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import to_onnx
from onnx import numpy_helper

from checkpoint import PROJECT_ROOT, make_manager, restore_checkpoint, save_checkpoint
from config import get_config
from model import ChessCNN
from selfplay.train import make_optimizer


#ONNX both ways: export the latest checkpoint, or import an .onnx file as a new checkpoint to keep training
#Run from src/: uv run python onnx_io.py export
#               uv run python onnx_io.py import path/to/file.onnx


#Writes the model with its current weights to one .onnx file (weights included, no .data sidecar)
#Input "board": float32 (batch, 8, 8, in_planes), "B" = any batch size; outputs "policy" and "value"
def export_onnx(model, path, in_planes: int):
    proto = to_onnx(model, [("B", 8, 8, in_planes)], input_names=["board"], output_names=["policy", "value"])
    onnx.save_model(proto, str(path), save_as_external_data=False)  # jax2onnx would split weights >1MB into .data
    return path


#Copies an .onnx file's weights into model, which must be built with the same config.model
#Weight names in the file are meaningless (const_6...), so layers are matched by graph order instead:
#Conv nodes = convs in order, Gemm (linear) nodes = policy head then value head, same order as __call__
def import_onnx(model, path):
    graph = onnx.load(str(path)).graph
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    convs = [node for node in graph.node if node.op_type == "Conv"]
    linears = [node for node in graph.node if node.op_type == "Gemm"]
    heads = list(model.policy_head) + list(model.value_head)
    if len(convs) != len(model.convs) or len(linears) != len(heads):
        raise ValueError(
            f"onnx has {len(convs)} conv / {len(linears)} linear layers, model has {len(model.convs)} / {len(heads)}"
            " - does config.model match the exported network?"
        )

    for node, conv in zip(convs, model.convs):
        _copy(conv.kernel, weights[node.input[1]].transpose(2, 3, 1, 0), node)  # onnx (out, in, h, w) -> flax (h, w, in, out)
        _copy(conv.bias, weights[node.input[2]], node)
    for node, linear in zip(linears, heads):
        kernel = weights[node.input[1]]
        if any(attr.name == "transB" and attr.i for attr in node.attribute):
            kernel = kernel.T  # Gemm can store the kernel flipped, flax wants (in, out)
        _copy(linear.kernel, kernel, node)
        _copy(linear.bias, weights[node.input[2]], node)


def _copy(param, array, node):
    if param.value.shape != array.shape:
        raise ValueError(f"{node.name}: onnx {array.shape} vs model {param.value.shape} - does config.model match?")
    param.value = jnp.asarray(array)


if __name__ == "__main__":
    config = get_config()
    model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, config.training)
    manager = make_manager(config.training)
    command = sys.argv[1:]

    if command == ["export"]:
        restore_checkpoint(manager, model, optimizer)  # random weights if nothing saved yet
        print("saved", export_onnx(model, PROJECT_ROOT / "chess.onnx", config.data.in_planes))
    elif len(command) == 2 and command[0] == "import":
        #Saved as the next iteration so training continues from it and older checkpoints stay as backups
        #AdamW starts fresh: an .onnx file only holds weights
        import_onnx(model, command[1])
        latest = manager.latest_step()
        iteration = 0 if latest is None else latest + 1
        save_checkpoint(manager, iteration, model, optimizer)
        print(f"imported {command[1]} as checkpoint {iteration}, make train continues from it")
    else:
        sys.exit("usage: python onnx_io.py export | python onnx_io.py import path/to/file.onnx")

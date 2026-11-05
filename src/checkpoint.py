from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx


#Checkpoints: model weights + AdamW state saved under the iteration number, last 3 kept
#Its own file so both selfplay (saving) and model.py (exporting) can import it without importing each other

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_manager(config) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        PROJECT_ROOT / config.checkpoint_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3)
    )


def save_checkpoint(manager, iteration: int, model, optimizer):
    state = {"model": nnx.to_pure_dict(nnx.state(model)), "optimizer": nnx.to_pure_dict(nnx.state(optimizer))}
    manager.save(iteration, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()


#Loads the latest checkpoint into model + optimizer, returns the iteration to continue from (0 if none)
def restore_checkpoint(manager, model, optimizer) -> int:
    latest = manager.latest_step()
    if latest is None:
        return 0
    model_state, optimizer_state = nnx.state(model), nnx.state(optimizer)
    target = {"model": nnx.to_pure_dict(model_state), "optimizer": nnx.to_pure_dict(optimizer_state)}
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, target)  # shapes only, tells orbax what to load
    restored = manager.restore(latest, args=ocp.args.StandardRestore(abstract))
    nnx.replace_by_pure_dict(model_state, restored["model"])
    nnx.replace_by_pure_dict(optimizer_state, restored["optimizer"])
    nnx.update(model, model_state)
    nnx.update(optimizer, optimizer_state)
    print(f"resumed from iteration {latest}")
    return latest + 1

import jax.numpy as jnp
import optax
from flax import nnx

from selfplay.loss import loss_fn as selfplay_loss


#Train step: one batch of positions -> loss -> gradients -> AdamW nudges the weights
#The model is updated in place, only the loss numbers come back (for printing)
#Shared with supervised/, which passes its own loss_fn; the targets are whatever that loss expects
#
#boards:        (batch, 8, 8, in_planes), encode_board output stacked
#policy_target: self-play: (batch, num_moves) float, MCTS visit percentages
#               supervised: (batch,) int, index of Stockfish's move
#value_target:  (batch,) float in [-1, 1], from the side to move's view

def make_optimizer(model, config) -> nnx.Optimizer:
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)  # wrt: only update the weights, nothing else


@nnx.jit(static_argnames=("value_weight", "loss_fn"))  # compiled once per loss, then fast on every call
def train_step(model, optimizer, boards, policy_target, value_target, value_weight: float = 1.0,
               loss_fn=selfplay_loss):
    def compute_loss(model):
        policy_logits, value_pred = model(boards.astype(jnp.float32))  # encoding is uint8, model wants floats
        return loss_fn(policy_logits, value_pred, policy_target, value_target, value_weight)

    #Loss and, for every weight, which direction makes the loss smaller
    (total, parts), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
    optimizer.update(model, grads)
    return total, parts


import jax.numpy as jnp
import optax


#Loss for the supervised jump start: how far the model is from Stockfish's move and eval
#Same idea as selfplay/loss.py, but the policy target is one move index instead of a spread of MCTS visits
#
#policy_logits: (batch, num_moves) float, raw move scores from the policy head
#value_pred:    (batch, 1) float, eval from the value head in [-1, 1]
#move_target:   (batch,) int, move_to_index of Stockfish's best move
#value_target:  (batch,) float, Stockfish's eval for the side to move squashed into [-1, 1] (see data/encode.py)
#
#Returns (total, parts): total is what backprop minimises, parts holds each piece for logging

def loss_fn(policy_logits, value_pred, move_target, value_target, value_weight: float = 1.0):
    #Policy: cross-entropy against the one right move, -log(softmax(logits)[move])
    #Same as a one-hot target in selfplay's loss, without building the 4672-wide one-hot
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(policy_logits, move_target).mean()

    #Value: mean squared error against Stockfish's eval, (prediction - eval)^2
    value_loss = jnp.mean((value_pred.squeeze(-1) - value_target) ** 2)  # squeeze: (batch, 1) -> (batch,)

    total = policy_loss + value_weight * value_loss
    return total, {"policy": policy_loss, "value": value_loss}


if __name__ == "__main__":
    #One board, pretend only 3 moves exist: Stockfish's move is index 0, its eval is +0.5
    total, parts = loss_fn(jnp.array([[2.0, 0.5, 0.1]]), jnp.array([[0.3]]), jnp.array([0]), jnp.array([0.5]))
    print(float(total), {k: float(v) for k, v in parts.items()})  # expect 0.36: policy 0.32 + value 0.04

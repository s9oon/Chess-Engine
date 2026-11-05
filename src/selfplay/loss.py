import jax.numpy as jnp
import optax


#Loss: how wrong the model was, as one number for backprop to minimise
#Knows nothing about the model or chess, it only compares predictions with self-play targets
#
#policy_logits: (batch, num_moves) float, raw move scores from the policy head
#value_pred:    (batch, 1) float, eval from the value head in [-1, 1]
#policy_target: (batch, num_moves) float, MCTS visit counts as probabilities (each row sums to 1)
#value_target:  (batch,) float, game result for the side to move: +1 win, 0 draw, -1 loss
#
#Returns (total, parts): total is what backprop minimises, parts holds each piece for logging

def loss_fn(policy_logits, value_pred, policy_target, value_target, value_weight: float = 1.0):
    #Policy: cross-entropy against the search's move spread, -sum(target * log(softmax(logits)))
    policy_loss = optax.softmax_cross_entropy(policy_logits, policy_target).mean()

    #Value: mean squared error against the game result, (prediction - result)^2
    value_loss = jnp.mean((value_pred.squeeze(-1) - value_target) ** 2)  # squeeze: (batch, 1) -> (batch,)

    total = policy_loss + value_weight * value_loss
    return total, {"policy": policy_loss, "value": value_loss}


if __name__ == "__main__":
    #One board, pretend only 3 moves exist: search visited them 60/30/10%, side to move went on to win
    total, parts = loss_fn(jnp.array([[2.0, 0.5, 0.1]]), jnp.array([[0.3]]), jnp.array([[0.6, 0.3, 0.1]]), jnp.array([1.0]))
    print(float(total), {k: float(v) for k, v in parts.items()})  # expect 1.45: policy 0.96 + value 0.49

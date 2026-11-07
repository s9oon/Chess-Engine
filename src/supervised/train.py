import multiprocessing
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from checkpoint import make_manager, restore_checkpoint, save_checkpoint
from config import get_config
from model import ChessCNN
from selfplay.train import make_optimizer, train_step
from supervised.data.decompress import open_evals, read_lines
from supervised.data.encode import encode_lines
from supervised.loss import loss_fn


#Supervised jump start: learn Stockfish's moves + evals from the Lichess evals file, before self-play
#Run from src/: uv run python -m supervised.train path/to/lichess_db_eval.jsonl.zst
#
#Loop: read X lines -> encode on the workers -> shuffle -> train in batches -> drop -> repeat until N
#Checkpoints go to checkpoints/supervised_<k> (k = validation checks done), apart from self-play's;
#make handoff copies the latest one over to self-play
#
#Re-running resumes: weights come from the latest supervised checkpoint and the lines it already
#trained on are skipped, so nothing is learned twice


#Validation: how the model does on positions it never trains on, without changing any weights
#accuracy = how often its top move is Stockfish's move
@nnx.jit(static_argnames="value_weight")
def evaluate(model, boards, moves, values, value_weight: float):
    policy_logits, value_pred = model(boards.astype(jnp.float32))
    _, parts = loss_fn(policy_logits, value_pred, moves, values, value_weight)
    accuracy = jnp.mean(jnp.argmax(policy_logits, axis=-1) == moves)
    return parts["policy"], parts["value"], accuracy


#Averages evaluate over the validation set in full batches (a short last batch would need a recompile)
def validate(model, val, config) -> tuple[float, float, float]:
    planes, moves, values = val
    batch = config.training.batch_size
    results = [
        evaluate(model, planes[i:i + batch], moves[i:i + batch], values[i:i + batch], config.training.value_weight)
        for i in range(0, len(moves) - batch + 1, batch)
    ]
    return tuple(float(x) for x in np.mean(results, axis=0))


def main(config, path: Path):
    sup = config.supervised
    rng = np.random.default_rng()

    #Workers are forked before the model exists: forking is fast and cheap, but only safe while
    #JAX hasn't started its threads, which happens once the model is built
    with multiprocessing.get_context("fork").Pool(sup.num_workers) as pool, open_evals(path) as file:
        model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
        optimizer = make_optimizer(model, config.training)
        manager = make_manager(config.training, prefix="supervised")
        check = restore_checkpoint(manager, model, optimizer)  # validation checks already done, 0 if new

        val = encode_lines(pool, read_lines(file, sup.val_positions), sup.cp_scale, rng)

        #Resuming: skip the lines earlier runs already trained on
        trained = check * sup.eval_every
        for _ in range(0, trained, sup.chunk_positions):
            read_lines(file, sup.chunk_positions)
        if trained:
            print(f"skipped {trained} already-trained positions")

        batch = config.training.batch_size
        next_check = trained + sup.eval_every
        start, start_trained = time.time(), trained

        while not sup.max_positions or trained < sup.max_positions:
            lines = read_lines(file, sup.chunk_positions)
            if not lines:
                break  # end of the file
            planes, moves, values = encode_lines(pool, lines, sup.cp_scale, rng)

            #Full batches only, the few leftover positions per chunk are dropped
            for i in range(0, len(moves) - batch + 1, batch):
                total, parts = train_step(
                    model, optimizer, planes[i:i + batch], moves[i:i + batch].astype(np.int32),
                    values[i:i + batch], config.training.value_weight, loss_fn=loss_fn,
                )
            trained += len(lines)
            print(f"{trained} positions ({(trained - start_trained) / (time.time() - start):.0f}/s): "
                  f"total {float(total):.3f}  policy {float(parts['policy']):.3f}  value {float(parts['value']):.3f}")

            if trained >= next_check:
                policy, value, accuracy = validate(model, val, config)
                print(f"validation: policy {policy:.3f}  value {value:.3f}  move accuracy {accuracy:.1%}")
                save_checkpoint(manager, check, model, optimizer)
                print(f"saved checkpoints/supervised_{check}")
                check += 1
                next_check += sup.eval_every

    print("done, run make handoff to continue from it with self-play")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m supervised.train path/to/lichess_db_eval.jsonl.zst")
    main(get_config(), Path(sys.argv[1]))

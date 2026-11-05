import time
from collections import deque
from itertools import count

import chess
import numpy as np
from flax import nnx

from checkpoint import make_manager, restore_checkpoint, save_checkpoint
from config import get_config
from encoding.board import encode_board
from model import ChessCNN
from selfplay.mcts import run_mcts
from selfplay.train import make_optimizer, train_step


#Self-play loop: play games -> replay buffer -> train on random batches -> checkpoint -> repeat
#Run from src/: uv run python -m selfplay.selfplay   (resumes from the latest checkpoint if there is one)


#One game against itself, returns its positions as (board planes, policy target, value target)
def play_game(model, config, rng: np.random.Generator) -> tuple[list, str]:
    board = chess.Board()
    history = []  # (planes, visit policy, side to move), result filled in once the game ends

    while board.outcome(claim_draw=True) is None and board.ply() < config.training.max_plies:
        tree = run_mcts(board, model, config.mcts, rng)
        history.append((encode_board(board), tree.visit_policy(), board.turn))
        #Early on pick by visit share so games differ, after that play the best move
        board.push(tree.choose_move(random=board.ply() < config.mcts.temperature_moves))

    outcome = board.outcome(claim_draw=True)
    winner = outcome.winner if outcome else None  # cut off at max_plies counts as a draw

    #Value target: +1 if the side to move in that position went on to win, -1 if it lost, 0 for a draw
    samples = [
        (planes, policy, 0.0 if winner is None else (1.0 if turn == winner else -1.0))
        for planes, policy, turn in history
    ]
    return samples, board.result(claim_draw=True) if outcome else "1/2-1/2 (cut off)"


#Random batch from the replay buffer, stacked into arrays for train_step
def sample_batch(buffer: deque, batch_size: int, rng: np.random.Generator):
    picks = rng.choice(len(buffer), size=batch_size, replace=False)
    planes, policies, values = zip(*(buffer[i] for i in picks))
    return np.stack(planes), np.stack(policies), np.array(values, dtype=np.float32)


def main(config):
    rng = np.random.default_rng()
    model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, config.training)

    manager = make_manager(config.training)
    start = restore_checkpoint(manager, model, optimizer)
    buffer = deque(maxlen=config.training.buffer_size)  # not checkpointed, refills after a restart

    for iteration in count(start):
        #Play
        for game in range(config.training.games_per_iteration):
            start_time = time.time()
            samples, result = play_game(model, config, rng)
            buffer.extend(samples)
            print(f"iter {iteration} game {game}: {result} in {len(samples)} plies, "
                  f"{time.time() - start_time:.0f}s, buffer {len(buffer)}")

        #Train, once there are enough positions for a full batch
        if len(buffer) >= config.training.batch_size:
            totals = []
            for _ in range(config.training.steps_per_iteration):
                boards, policy_target, value_target = sample_batch(buffer, config.training.batch_size, rng)
                total, parts = train_step(
                    model, optimizer, boards, policy_target, value_target, config.training.value_weight
                )
                totals.append((float(total), float(parts["policy"]), float(parts["value"])))
            total, policy, value = np.mean(totals, axis=0)
            print(f"iter {iteration} train: total {total:.3f}  policy {policy:.3f}  value {value:.3f}")

        if iteration % config.training.checkpoint_every == 0:
            save_checkpoint(manager, iteration, model, optimizer)


if __name__ == "__main__":
    main(get_config())

# Chess Bot v2

AlphaZero-style chess bot written in JAX/Flax. It learns only by playing against itself: no human games and no Stockfish data.

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

| command | what it does |
|---|---|
| `make train` | self-play + training, resumes from the latest checkpoint (Ctrl+C to stop) |
| `make serve` | move API on `http://localhost:8000` |
| `make export` | latest checkpoint -> `chess.onnx` |
| `make import ONNX=file.onnx` | `.onnx` file -> new checkpoint, `make train` continues from it |

Checkpoints are saved to `checkpoints/`, which keeps the last 3. Back this folder up, because it is the trained bot.

## Move API

```
POST /move
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}

-> {"uci": "e2e4", "san": "e4", "eval": 0.12}
```

`eval` is how good the move looks to the bot, from -1 (losing) to +1 (winning) for the side to move. An invalid FEN or a finished game returns 400. Interactive docs are at `/docs`.

The server loads the latest checkpoint at startup, so restart it after training to use the new weights.

## How it works

1. **Self-play:** for each move, MCTS searches ahead using the network and records how often it visited each move.
2. **Targets:** the visit percentages become the policy target, and the game result (+1 / 0 / -1) becomes the value target.
3. **Training:** the policy is trained with cross-entropy against the search and the value with MSE against the result, using AdamW.
4. Repeat. A better network gives a better search, which gives better targets.

## Layout

```
src/
  config.py          all settings (model, MCTS, training)
  model.py           CNN with policy (4672 moves) and value heads
  encoding/
    board.py         board -> (8, 8, 18) planes
    policy.py        move <-> policy index (from_square * 73 + move_type)
  selfplay/
    mcts.py          tree search
    loss.py          policy cross-entropy + value MSE
    train.py         AdamW train step
    selfplay.py      game loop, replay buffer, checkpoints
  checkpoint.py      save / resume with orbax
  server.py          FastAPI move server
  onnx_io.py         ONNX export / import
```

## Status

The pipeline runs end to end, but the bot has not been trained long enough to know how strong it gets. Next steps:

- strength check against a random mover and against older checkpoints
- batch self-play across many games, because one game at a time is slow

## License

MIT, see [LICENSE](LICENSE).

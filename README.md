# Chess Bot v2

AlphaZero-style chess bot written in JAX/Flax. It learns by playing against itself. There is an optional jump start: pretrain on Stockfish evaluations from Lichess first, so self-play doesn't begin from random weights.

## Repo layout

| folder | what it is |
|---|---|
| `engine/` | Python project: the model, training (self-play + optional pretraining), ONNX export |
| `web/backend/` | ASP.NET inference API: loads `chess.onnx`, serves moves (in progress) |
| `web/frontend/` | frontend, planned (empty for now) |

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything below runs from `engine/`.

```bash
cd engine
uv sync
```

## Usage

| command | what it does |
|---|---|
| `make train` | self-play + training, resumes from the latest checkpoint (Ctrl+C to stop) |
| `make serve` | move API on `http://localhost:8000` |
| `make export` | latest checkpoint -> `chess.onnx` |
| `make import ONNX=file.onnx` | `.onnx` file -> new checkpoint, `make train` continues from it |
| `make pretrain EVALS=file.jsonl.zst` | optional: supervised pretraining on Lichess evals, resumes if run again |
| `make handoff` | latest pretraining checkpoint -> next self-play checkpoint, `make train` continues from it |

Checkpoints are saved to `engine/checkpoints/`: self-play as `28`, `29`, ... and pretraining as `supervised_0`, `supervised_1`, .... Each keeps its own last 3. Back this folder up, because it is the trained bot.

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

### Optional: pretraining on Stockfish evals

Self-play from random weights is slow to get going, so the network can first learn from the [Lichess evals database](https://database.lichess.org/#evals) (`lichess_db_eval.jsonl.zst`, about 22.5 GB, over 400M positions evaluated by Stockfish).

1. **Read:** the file is decompressed as it is read, one chunk of positions at a time (200k by default), never unpacked whole.
2. **Encode:** worker processes turn each position into board planes plus two targets taken from the deepest evaluation's best line. The policy target is its first move, and the value target is its score for the side to move, squashed with `tanh(cp / 400)` (a forced mate counts as +1 or -1).
3. **Train:** each chunk is shuffled and trained on in batches with cross-entropy on the move and MSE on the value. Every 5M positions it reports validation loss and move accuracy (how often its top move matches Stockfish's) and saves `checkpoints/supervised_<k>`. It stops after 50M positions by default.
4. **Hand off:** `make handoff` copies the latest pretraining checkpoint to the next self-play checkpoint. From there, self-play takes over.

The settings are in the `supervised` section of `config.py`.

## Engine layout

```
engine/
  Makefile, pyproject.toml, uv.lock
  checkpoints/         saved models (not in git)
  training/
    config.py          all settings (model, MCTS, training, pretraining)
    model.py           CNN with policy (4672 moves) and value heads
    encoding/
      board.py         board -> (8, 8, 18) planes
      policy.py        move <-> policy index (from_square * 73 + move_type)
    selfplay/
      mcts.py          tree search
      loss.py          policy cross-entropy + value MSE
      train.py         AdamW train step (also used by pretraining)
      selfplay.py      game loop, replay buffer, checkpoints
    supervised/        optional pretraining on Lichess Stockfish evals
      data/
        decompress.py  read the .zst file a chunk of lines at a time
        encode.py      JSON line -> board planes, move index, value (on worker processes)
      loss.py          policy cross-entropy on the move index + value MSE
      train.py         read -> encode -> train loop, validation, checkpoints
      handoff.py       latest pretraining checkpoint -> next self-play checkpoint
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

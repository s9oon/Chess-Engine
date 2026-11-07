import json
from functools import partial
from multiprocessing.pool import Pool

import chess
import numpy as np

from encoding.board import encode_board
from encoding.policy import move_to_index


#Lichess eval lines -> training arrays, split across worker processes
#Must not import jax: the workers are spawned and import this file, they only need chess + numpy
#
#One line looks like:
#{"fen": "...", "evals": [{"depth": 36, "pvs": [{"cp": 311, "line": "g2e4 f7f5 ..."}, ...]}, ...]}
#
#Labels come from the deepest eval's best line (what Lichess recommends):
#policy: its first move, as a move_to_index slot
#value:  its score for the side to move, squashed into [-1, 1]; cp/mate in the file are from white's view


#One JSON line -> (board planes, move index, value)
def encode_line(line: str, cp_scale: float) -> tuple[np.ndarray, int, float]:
    position = json.loads(line)
    board = chess.Board(position["fen"])
    best = max(position["evals"], key=lambda e: e["depth"])["pvs"][0]

    move = board.parse_uci(best["line"].split()[0])  # also turns Lichess castling (e1h1) into e1g1

    if "mate" in best:
        value = float(np.sign(best["mate"]))  # mate for white = +1, mate for black = -1
    else:
        value = float(np.tanh(best["cp"] / cp_scale))  # +cp_scale centipawns ~ 0.76
    if board.turn == chess.BLACK:
        value = -value  # white's view -> side to move's view, same as self-play's targets

    return encode_board(board), move_to_index(move), value


#Many lines -> three arrays, encoded by the pool's workers, then shuffled
#planes: (n, 8, 8, 18) uint8 | moves: (n,) int16 | values: (n,) float32
def encode_lines(pool: Pool, lines: list[str], cp_scale: float, rng: np.random.Generator):
    rows = pool.map(partial(encode_line, cp_scale=cp_scale), lines)  # map hands lines out in batches
    planes, moves, values = zip(*rows)

    order = rng.permutation(len(lines))  # nearby lines in the file can be similar, so mix them
    planes = np.stack(planes)[order]
    moves = np.array(moves, dtype=np.int16)[order]
    values = np.array(values, dtype=np.float32)[order]
    return planes, moves, values

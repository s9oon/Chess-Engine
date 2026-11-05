import math

import chess
import jax.numpy as jnp
import numpy as np
from flax import nnx

from encoding.board import encode_board
from encoding.policy import NUM_MOVES, legal_mask, move_to_index


#MCTS: thinks ahead from one position and returns how often it visited each move (the policy target)
#
#The tree never calls the model itself. One simulation is:
#   board = tree.select_leaf()                     walk down to a position nobody has evaluated yet
#   priors, values = evaluate(model, [board])      network call, takes a batch (1 board now, many later)
#   tree.expand_and_backup(priors[0], values[0])   add the leaf's children, push its value back up
#So batching many games later = many trees, one evaluate call, no changes in here
#
#Values are always for the side to move, same as the value head and value_target


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior  # policy head's score for the move leading here
        self.visits = 0
        self.value_sum = 0.0  # from the view of the player who made the move leading here
        self.children = {}  # chess.Move -> Node, empty until expanded

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class Tree:
    def __init__(self, board: chess.Board, config, rng: np.random.Generator | None = None):
        if board.is_game_over(claim_draw=True):
            raise ValueError("can't search a finished game")
        self.board = board.copy()
        self.root = Node(prior=1.0)
        self.config = config
        self.rng = rng  # None = no root noise (for playing, not self-play)
        self._path = []
        self._leaf_board = None

    #Walk down picking the best-scoring child until reaching an unexpanded node
    #Returns the leaf's board for the network, or None if the leaf was a finished game (already backed up)
    def select_leaf(self) -> chess.Board | None:
        node = self.root
        board = self.board.copy()
        path = [node]
        while node.children:
            node_visits_sqrt = math.sqrt(node.visits)
            best_score, best_move, best_child = -math.inf, None, None
            for move, child in node.children.items():
                #PUCT: how good it looks + bonus for high prior and few visits
                score = child.q() + self.config.c_puct * child.prior * node_visits_sqrt / (1 + child.visits)
                if score > best_score:
                    best_score, best_move, best_child = score, move, child
            board.push(best_move)
            node = best_child
            path.append(node)

        self._path = path
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            #Finished game: real result instead of the network, checkmate means side to move lost
            self._backup(0.0 if outcome.winner is None else -1.0)
            return None
        self._leaf_board = board
        return board

    #priors: (num_moves,) probabilities over legal moves, value: leaf eval for its side to move
    def expand_and_backup(self, priors: np.ndarray, value: float):
        leaf = self._path[-1]
        for move in self._leaf_board.legal_moves:
            leaf.children[move] = Node(prior=float(priors[move_to_index(move)]))
        if leaf is self.root and self.rng is not None:
            self._add_root_noise()
        self._backup(float(value))

    #Mix random noise into the root's priors so self-play explores moves the network currently ignores
    def _add_root_noise(self):
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(self.root.children))
        eps = self.config.dirichlet_epsilon
        for child, n in zip(self.root.children.values(), noise):
            child.prior = (1 - eps) * child.prior + eps * n

    #value is for the side to move at the leaf, flip it at every level going up
    def _backup(self, value: float):
        for node in reversed(self._path):
            value = -value  # node stores it from the view of whoever moved into it
            node.visits += 1
            node.value_sum += value

    #Root visit counts as percentages over all moves: this is the policy_target
    def visit_policy(self) -> np.ndarray:
        policy = np.zeros(NUM_MOVES, dtype=np.float32)
        for move, child in self.root.children.items():
            policy[move_to_index(move)] = child.visits
        return policy / policy.sum()

    #random=True: pick in proportion to visits (early self-play moves), False: most visited
    def choose_move(self, random: bool) -> chess.Move:
        moves = list(self.root.children)
        visits = np.array([self.root.children[m].visits for m in moves], dtype=np.float64)
        if random:
            return moves[self.rng.choice(len(moves), p=visits / visits.sum())]
        return moves[int(np.argmax(visits))]


@nnx.jit  # compiled once per batch size
def _forward(model, x):
    return model(x)


#Network call for a batch of boards -> priors (batch, num_moves) over legal moves only, values (batch,)
def evaluate(model, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
    x = jnp.asarray(np.stack([encode_board(b) for b in boards]), dtype=jnp.float32)
    logits, values = _forward(model, x)
    logits = np.where(np.stack([legal_mask(b) for b in boards]), np.asarray(logits), -np.inf)
    priors = np.exp(logits - logits.max(axis=1, keepdims=True))  # softmax over legal moves only
    return priors / priors.sum(axis=1, keepdims=True), np.asarray(values)[:, 0]


#Single game, batch of 1: full search from one position
def run_mcts(board: chess.Board, model, config, rng: np.random.Generator | None = None) -> Tree:
    tree = Tree(board, config, rng)
    for _ in range(config.num_simulations):
        leaf = tree.select_leaf()
        if leaf is not None:
            priors, values = evaluate(model, [leaf])
            tree.expand_and_backup(priors[0], values[0])
    return tree

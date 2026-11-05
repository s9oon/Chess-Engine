import chess
import numpy as np


#Policy output: 4672 move scores, index = from_square * 73 + move_type
#from_square uses python-chess numbering: 0 = a1, 7 = h1, 8 = a2 ... 63 = h8
#
#move_type 0-55:  queen-style slides, direction * 7 + (distance - 1)
#                 covers rook/bishop/queen/king moves, pawn pushes + captures, castling, queen promotions
#move_type 56-63: knight jumps
#move_type 64-72: underpromotions, 64 + direction * 3 + piece
#                 direction: 0 = capture towards a-file, 1 = straight, 2 = capture towards h-file
#                 piece: 0 = knight, 1 = bishop, 2 = rook
#
#Most indices are never legal (e.g. sliding north off the board), legal_mask filters them out

NUM_MOVE_TYPES = 73
NUM_MOVES = 64 * NUM_MOVE_TYPES  # 4672, must match config.data.num_moves

#(file step, rank step) per direction: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
KNIGHT_JUMPS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


#Move -> policy index: turns a training label (the move played) into the slot the loss should push up
def move_to_index(move: chess.Move) -> int:
    file_step = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    rank_step = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    if move.promotion in UNDERPROMOTIONS:
        move_type = 64 + (file_step + 1) * 3 + UNDERPROMOTIONS.index(move.promotion)
    elif (file_step, rank_step) in KNIGHT_JUMPS:
        move_type = 56 + KNIGHT_JUMPS.index((file_step, rank_step))
    else:
        #Queen-style: split the step into a direction and a distance (1-7)
        distance = max(abs(file_step), abs(rank_step))
        direction = QUEEN_DIRECTIONS.index((file_step // distance, rank_step // distance))
        move_type = direction * 7 + (distance - 1)

    return move.from_square * NUM_MOVE_TYPES + move_type


#Policy index -> move: turns the model's chosen slot back into a move to play
#Needs the board only to tell if a queen-style pawn move to the last rank is a queen promotion
def index_to_move(index: int, board: chess.Board) -> chess.Move:
    from_square, move_type = divmod(index, NUM_MOVE_TYPES)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    promotion = None

    if move_type >= 64:
        direction, piece = divmod(move_type - 64, 3)
        file_step = direction - 1
        rank_step = 1 if from_rank == 6 else -1  # white promotes from rank 7, black from rank 2
        promotion = UNDERPROMOTIONS[piece]
    elif move_type >= 56:
        file_step, rank_step = KNIGHT_JUMPS[move_type - 56]
    else:
        direction, distance = divmod(move_type, 7)
        file_step, rank_step = QUEEN_DIRECTIONS[direction]
        file_step *= distance + 1
        rank_step *= distance + 1

    to_file, to_rank = from_file + file_step, from_rank + rank_step
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError(f"index {index} moves off the board")

    if promotion is None and board.piece_type_at(from_square) == chess.PAWN and to_rank in (0, 7):
        promotion = chess.QUEEN

    return chess.Move(from_square, chess.square(to_file, to_rank), promotion)


#True for every slot that is a legal move in this position, False everywhere else
def legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


#Policy head output (4672 scores for one board) -> best legal move
def policy_to_move(policy, board: chess.Board) -> chess.Move:
    if board.is_game_over():
        raise ValueError("no legal moves, the game is over")
    scores = np.where(legal_mask(board), np.asarray(policy), -np.inf)  # illegal slots can never win
    return index_to_move(int(np.argmax(scores)), board)


if __name__ == "__main__":
    #Round trip check: play random games, every legal move must survive move -> index -> move
    #and no two legal moves in a position may share an index
    import random
    random.seed(0)
    checked = 0
    for _ in range(200):
        board = chess.Board()
        while not board.is_game_over():
            legal = list(board.legal_moves)
            indices = [move_to_index(m) for m in legal]
            assert len(set(indices)) == len(legal), board.fen()
            for move, index in zip(legal, indices):
                assert index_to_move(index, board) == move, (board.fen(), move)
            assert legal_mask(board).sum() == len(legal)
            checked += len(legal)
            board.push(random.choice(legal))

    #Promotions and castling, which random games rarely reach
    board = chess.Board("r3k2r/1P6/8/8/8/8/1p6/R3K2R w KQkq - 0 1")
    for move in board.legal_moves:
        assert index_to_move(move_to_index(move), board) == move, move
    print(f"round trip ok on {checked} moves")
    print("best move from random scores:", policy_to_move(np.random.randn(NUM_MOVES), chess.Board()))

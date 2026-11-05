import chess
import numpy as np


#Input feature map: board -> (8, 8, 18) planes, 1 byte (uint8, 0 or 1) per square per plane
#Indexed planes[rank, file, plane]; rank 0 = rank 1 (white's side), file 0 = the a-file
#
#planes 0-5:   white pawn, knight, bishop, rook, queen, king
#planes 6-11:  black pawn, knight, bishop, rook, queen, king
#plane 12:     side to move (all 1s if white)
#planes 13-16: castling rights (white kingside, white queenside, black kingside, black queenside)
#plane 17:     en passant target square

def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 18), dtype=np.uint8)

    #Pieces: one plane per piece type and colour
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        plane = piece.piece_type - 1  # pawn = 1 ... king = 6, so 0-5
        if piece.color == chess.BLACK:
            plane += 6  # black pieces go on planes 6-11
        planes[rank, file, plane] = 1

    #Side to move: whole plane so every conv patch can see it
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1

    #Castling rights: whole planes, same reason
    planes[:, :, 13] = board.has_kingside_castling_rights(chess.WHITE)
    planes[:, :, 14] = board.has_queenside_castling_rights(chess.WHITE)
    planes[:, :, 15] = board.has_kingside_castling_rights(chess.BLACK)
    planes[:, :, 16] = board.has_queenside_castling_rights(chess.BLACK)

    #En passant: just the target square, if there is one
    if board.ep_square is not None:
        planes[chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 17] = 1

    return planes


if __name__ == "__main__":
    print(encode_board(chess.Board())[:, :, 0])  # white pawns: row 1 (rank 2) should be all 1s

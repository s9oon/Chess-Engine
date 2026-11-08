namespace backend.Chess;

/// <summary>
/// FEN string -> model input, matching engine/training/encoding/board.py (encode_board).
///
/// Layout is planes[rank, file, plane] flattened row-major into 8 * 8 * 18 floats:
///   index = (rank * 8 + file) * 18 + plane
///   rank 0 = rank 1 (White's side), file 0 = the a-file
///
/// planes 0-5:   white pawn, knight, bishop, rook, queen, king
/// planes 6-11:  black pawn, knight, bishop, rook, queen, king
/// plane 12:     side to move (all 1s if white)
/// planes 13-16: castling rights (white kingside, white queenside, black kingside, black queenside)
/// plane 17:     en passant target square
///
/// Expects a well-formed FEN (FenRequest's [Fen] attribute checks that before this runs).
/// </summary>
public static class FenEncoder
{
    public const int NumPlanes = 18;
    public const int InputSize = 8 * 8 * NumPlanes; // 1152

    // Same order as python-chess piece types: pawn = 1 ... king = 6, so index 0-5
    private const string PieceOrder = "PNBRQK";

    public static float[] Encode(string fen)
    {
        float[] planes = new float[InputSize];
        string[] fields = fen.Split(' ');

        // Pieces: FEN lists rank 8 first, so the i-th FEN rank is board rank 7 - i
        string[] ranks = fields[0].Split('/');
        for (int i = 0; i < 8; i++)
        {
            int rank = 7 - i;
            int file = 0;

            foreach (char c in ranks[i])
            {
                if (char.IsDigit(c))
                {
                    file += c - '0'; // digit = that many empty squares
                    continue;
                }

                int plane = PieceOrder.IndexOf(char.ToUpperInvariant(c));
                if (char.IsLower(c))
                    plane += 6; // black pieces go on planes 6-11

                planes[Index(rank, file, plane)] = 1f;
                file++;
            }
        }

        // Side to move: whole plane so every conv patch can see it
        if (fields[1] == "w")
            FillPlane(planes, 12);

        // Castling rights: whole planes, same reason
        string castling = fields[2];
        if (castling.Contains('K')) FillPlane(planes, 13);
        if (castling.Contains('Q')) FillPlane(planes, 14);
        if (castling.Contains('k')) FillPlane(planes, 15);
        if (castling.Contains('q')) FillPlane(planes, 16);

        // En passant: just the target square, if there is one
        string enPassant = fields[3];
        if (enPassant != "-")
        {
            int file = enPassant[0] - 'a';
            int rank = enPassant[1] - '1';
            planes[Index(rank, file, 17)] = 1f;
        }

        return planes;
    }

    private static int Index(int rank, int file, int plane) => (rank * 8 + file) * NumPlanes + plane;

    private static void FillPlane(float[] planes, int plane)
    {
        for (int square = 0; square < 64; square++)
            planes[square * NumPlanes + plane] = 1f;
    }
}

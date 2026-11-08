namespace backend.Chess;

/// <summary>
/// Policy output (4672 scores) -> move, matching engine/training/encoding/policy.py.
///
/// index = from_square * 73 + move_type
/// from_square uses python-chess numbering: 0 = a1, 7 = h1, 8 = a2 ... 63 = h8
///
/// move_type 0-55:  queen-style slides, direction * 7 + (distance - 1)
///                  covers rook/bishop/queen/king moves, pawn pushes + captures, castling, queen promotions
/// move_type 56-63: knight jumps
/// move_type 64-72: underpromotions, 64 + direction * 3 + piece
///                  direction: 0 = capture towards a-file, 1 = straight, 2 = capture towards h-file
///                  piece: 0 = knight, 1 = bishop, 2 = rook
///
/// Moves are UCI strings ("e2e4", "e7e8q"), the same format python-chess's Move.uci() gives.
/// </summary>
public static class MoveDecoder
{
    public const int NumMoveTypes = 73;
    public const int NumMoves = 64 * NumMoveTypes; // 4672, must match the model's policy output

    // Same order as policy.py, any change here scrambles every move.
    // (file step, rank step) per direction: N, NE, E, SE, S, SW, W, NW
    private static readonly (int File, int Rank)[] QueenDirections =
        [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

    private static readonly (int File, int Rank)[] KnightJumps =
        [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

    private static readonly char[] Underpromotions = ['n', 'b', 'r'];

    /// <summary>
    /// Best move for this position according to the policy scores.
    ///
    /// With legalMoves (UCI strings from a move generator) this matches policy.py's policy_to_move exactly.
    /// Without it there is no move generator yet, so only a basic filter is applied: the move must stay on the
    /// board, start on one of the side to move's pieces, and not land on its own piece. That still allows
    /// moves that are illegal (wrong piece geometry, moving into check), so treat it as a stopgap.
    /// </summary>
    public static string Decode(ReadOnlySpan<float> policy, string fen, IReadOnlyCollection<string>? legalMoves = null)
    {
        if (policy.Length != NumMoves)
            throw new ArgumentException($"Expected {NumMoves} policy scores, got {policy.Length}.", nameof(policy));

        return legalMoves is null
            ? DecodeUnmasked(policy, fen)
            : DecodeLegal(policy, legalMoves);
    }

    // Highest-scoring move out of a known legal list; ties go to the lowest index, like np.argmax
    private static string DecodeLegal(ReadOnlySpan<float> policy, IReadOnlyCollection<string> legalMoves)
    {
        if (legalMoves.Count == 0)
            throw new InvalidOperationException("No legal moves, the game is over.");

        string best = "";
        int bestIndex = int.MaxValue;
        float bestScore = float.NegativeInfinity;

        foreach (string move in legalMoves)
        {
            int index = MoveToIndex(move);
            float score = policy[index];
            if (score > bestScore || (score == bestScore && index < bestIndex))
            {
                best = move;
                bestIndex = index;
                bestScore = score;
            }
        }

        return best;
    }

    // Stopgap until there is a move generator, see Decode
    private static string DecodeUnmasked(ReadOnlySpan<float> policy, string fen)
    {
        char[] board = ParsePlacement(fen);
        bool whiteToMove = fen.Split(' ')[1] == "w";

        string? best = null;
        float bestScore = float.NegativeInfinity;

        for (int index = 0; index < NumMoves; index++)
        {
            if (policy[index] <= bestScore)
                continue; // can't beat the current best, skip decoding it

            string? move = IndexToMove(index, board);
            if (move is null)
                continue; // slides off the board

            char moving = board[index / NumMoveTypes];
            char target = board[ParseSquare(move, 2)];
            if (moving == '\0' || IsWhite(moving) != whiteToMove)
                continue; // no piece of ours on the from-square
            if (target != '\0' && IsWhite(target) == whiteToMove)
                continue; // would capture our own piece

            best = move;
            bestScore = policy[index];
        }

        return best ?? throw new InvalidOperationException("No move found for this position.");
    }

    /// <summary>Move -> policy index, port of policy.py's move_to_index.</summary>
    public static int MoveToIndex(string uci)
    {
        int from = ParseSquare(uci, 0);
        int to = ParseSquare(uci, 2);
        int fileStep = to % 8 - from % 8;
        int rankStep = to / 8 - from / 8;
        int underpromotion = uci.Length == 5 ? Array.IndexOf(Underpromotions, uci[4]) : -1;

        int moveType;
        int knight = Array.IndexOf(KnightJumps, (fileStep, rankStep));

        if (underpromotion >= 0)
        {
            moveType = 64 + (fileStep + 1) * 3 + underpromotion;
        }
        else if (knight >= 0)
        {
            moveType = 56 + knight;
        }
        else
        {
            // Queen-style: split the step into a direction and a distance (1-7)
            int distance = Math.Max(Math.Abs(fileStep), Math.Abs(rankStep));
            int direction = distance == 0 ? -1 : Array.IndexOf(QueenDirections, (fileStep / distance, rankStep / distance));
            if (direction < 0)
                throw new ArgumentException($"'{uci}' is not a queen, knight or promotion move.", nameof(uci));
            moveType = direction * 7 + (distance - 1);
        }

        return from * NumMoveTypes + moveType;
    }

    /// <summary>Policy index -> move, port of policy.py's index_to_move. Null if it moves off the board.</summary>
    public static string? IndexToMove(int index, string fen) => IndexToMove(index, ParsePlacement(fen));

    private static string? IndexToMove(int index, char[] board)
    {
        int from = index / NumMoveTypes;
        int moveType = index % NumMoveTypes;
        int fromFile = from % 8;
        int fromRank = from / 8;
        char? promotion = null;
        int fileStep, rankStep;

        if (moveType >= 64)
        {
            int direction = (moveType - 64) / 3;
            int piece = (moveType - 64) % 3;
            fileStep = direction - 1;
            rankStep = fromRank == 6 ? 1 : -1; // white promotes from rank 7, black from rank 2
            promotion = Underpromotions[piece];
        }
        else if (moveType >= 56)
        {
            (fileStep, rankStep) = KnightJumps[moveType - 56];
        }
        else
        {
            int direction = moveType / 7;
            int distance = moveType % 7 + 1;
            (fileStep, rankStep) = QueenDirections[direction];
            fileStep *= distance;
            rankStep *= distance;
        }

        int toFile = fromFile + fileStep;
        int toRank = fromRank + rankStep;
        if (toFile is < 0 or > 7 || toRank is < 0 or > 7)
            return null;

        // A pawn reaching the last rank with a queen-style move is a queen promotion
        if (promotion is null && char.ToLowerInvariant(board[from]) == 'p' && toRank is 0 or 7)
            promotion = 'q';

        return $"{SquareName(from)}{SquareName(toRank * 8 + toFile)}{promotion}";
    }

    // FEN placement -> 64 squares (0 = a1), piece letter or '\0' for empty
    private static char[] ParsePlacement(string fen)
    {
        char[] board = new char[64];
        string[] ranks = fen.Split(' ')[0].Split('/');

        for (int i = 0; i < 8; i++)
        {
            int rank = 7 - i; // FEN lists rank 8 first
            int file = 0;
            foreach (char c in ranks[i])
            {
                if (char.IsDigit(c))
                    file += c - '0';
                else
                    board[rank * 8 + file++] = c;
            }
        }

        return board;
    }

    private static bool IsWhite(char piece) => char.IsUpper(piece);

    private static int ParseSquare(string uci, int offset) => (uci[offset + 1] - '1') * 8 + (uci[offset] - 'a');

    private static string SquareName(int square) => $"{(char)('a' + square % 8)}{(char)('1' + square / 8)}";
}

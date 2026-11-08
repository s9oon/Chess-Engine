using System.ComponentModel.DataAnnotations;
using System.Text.RegularExpressions;

namespace backend.Dtos;

/// <summary>
/// Validates that a string is well-formed FEN (Forsyth-Edwards Notation).
/// Checks syntax only, not whether the position is legal chess.
/// </summary>
public sealed class FenAttribute : ValidationAttribute
{
    // Six space-separated fields:
    // 1. placement  2. side to move  3. castling  4. en passant  5. halfmove  6. fullmove
    private static readonly Regex Shape = new(
        @"^([pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+ [wb] (-|(?=[KQkq])K?Q?k?q?) (-|[a-h][36]) \d+ [1-9]\d*$",
        RegexOptions.Compiled);

    protected override ValidationResult? IsValid(object? value, ValidationContext context)
    {
        if (value is not string fen || !Shape.IsMatch(fen))
            return new ValidationResult("Not a valid FEN string.");

        string[] fields = fen.Split(' ');

        // Field 1: each rank must cover exactly 8 squares.
        foreach (string rank in fields[0].Split('/'))
        {
            int squares = 0;
            foreach (char c in rank)
                squares += char.IsDigit(c) ? c - '0' : 1;

            if (squares != 8)
                return new ValidationResult($"Rank '{rank}' does not cover 8 squares.");
        }

        // Field 4: en passant square must be on rank 6 if White moves, rank 3 if Black moves.
        string sideToMove = fields[1];
        string enPassant = fields[3];
        if (enPassant != "-")
        {
            char expectedRank = sideToMove == "w" ? '6' : '3';
            if (enPassant[1] != expectedRank)
                return new ValidationResult(
                    $"En passant square '{enPassant}' must be on rank {expectedRank} when '{sideToMove}' is to move.");
        }

        return ValidationResult.Success;
    }
}

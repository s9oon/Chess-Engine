using System.ComponentModel.DataAnnotations;
using backend.Chess;

namespace backend.Dtos;

public record FenRequest
{
    [Required, Fen]
    public required string Fen { get; init; }
    public ChessModel Model { get; init; } = ChessModel.Goataro;
}

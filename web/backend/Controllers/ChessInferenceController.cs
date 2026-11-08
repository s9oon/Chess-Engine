using Microsoft.AspNetCore.Mvc;
using backend.Chess;
using backend.Dtos;

namespace backend.Controllers;

[ApiController]
[Route("[controller]")]
public class ChessInferenceController : ControllerBase
{
    private readonly Inference _inference;

    public ChessInferenceController(Inference inference)
    {
        _inference = inference;
    }

    [HttpPost("move")]                                     // POST /ChessInference/move
    public IActionResult GetMove([FromBody] FenRequest request)
    {
        return Ok(_inference.GetMove(request.Fen, request.Model));
    }
}
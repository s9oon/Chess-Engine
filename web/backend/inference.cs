namespace backend.Chess;

using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Reflection.Metadata;
using backend.Chess;

public class Inference
{
    private readonly Dictionary<ChessModel, InferenceSession> _sessions;

    public Inference(Dictionary<ChessModel, InferenceSession> sessions)
    {
        _sessions = sessions;
    }

    public MoveResult GetMove(string fen, ChessModel model)
    {
        InferenceSession session = _sessions[model];
        float[] planes = FenEncoder.Encode(fen);

        // Wrap the flat array as a (batch, rank, file, plane) tensor; no copy, just a shape
        using var input = OrtValue.CreateTensorValueFromMemory(planes, new long[] { 1, 8, 8, FenEncoder.NumPlanes });
        var inputs = new Dictionary<string, OrtValue> { { "board", input } }; // name from onnx_io.py export

        using var runOptions = new RunOptions();
        using var outputs = session.Run(runOptions, inputs, session.OutputNames);

        // Outputs come back in session.OutputNames order: "policy" then "value"
        ReadOnlySpan<float> policy = outputs[0].GetTensorDataAsSpan<float>(); // 4672 move scores
        float eval = outputs[1].GetTensorDataAsSpan<float>()[0];               // -1 .. 1, side to move

        string move = MoveDecoder.Decode(policy, fen);
        return new MoveResult(move, eval);
    }
}

public record MoveResult(string Move, float Eval);
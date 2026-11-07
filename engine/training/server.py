import chess
from fastapi import FastAPI, HTTPException
from flax import nnx
from pydantic import BaseModel

from checkpoint import make_manager, restore_checkpoint
from config import get_config
from model import ChessCNN
from selfplay.mcts import run_mcts
from selfplay.train import make_optimizer


#Move server: POST /move with a FEN, the bot searches with MCTS and answers with its move
#Run from engine/training/: uv run fastapi run server.py   (loads the latest checkpoint at startup)

config = get_config()
model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
restore_checkpoint(make_manager(config.training), model, make_optimizer(model, config.training))  # optimizer needed to load, then unused

app = FastAPI()


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    uci: str  # e.g. "e2e4", "e7e8q"
    san: str  # e.g. "e4", "e8=Q"
    eval: float  # how good the move looks to the bot, -1 losing to +1 winning, for the side to move


#Plain def (not async): FastAPI runs it in a worker thread so a long search doesn't block other requests
@app.post("/move")
def move(request: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(400, "invalid FEN")
    if board.is_game_over(claim_draw=True):
        raise HTTPException(400, f"game is over: {board.result(claim_draw=True)}")

    tree = run_mcts(board, model, config.mcts)  # no rng: no noise, always the best move
    best = tree.choose_move(random=False)
    return MoveResponse(uci=best.uci(), san=board.san(best), eval=tree.root.children[best].q())

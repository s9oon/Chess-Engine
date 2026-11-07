import sys

from flax import nnx

from checkpoint import make_manager, restore_checkpoint, save_checkpoint
from config import get_config
from model import ChessCNN
from selfplay.train import make_optimizer


#Handoff: latest checkpoints/supervised_<k> -> the next self-play checkpoint, so make train continues from it
#Older self-play checkpoints stay as backups until self-play's own saves push them out (last 3 kept)
#Run from src/: uv run python -m supervised.handoff

def handoff(config):
    model = ChessCNN(config.model, config.data.in_planes, config.data.num_moves, rngs=nnx.Rngs(0))
    optimizer = make_optimizer(model, config.training)

    supervised = make_manager(config.training, prefix="supervised")
    if supervised.latest_step() is None:
        sys.exit("no supervised checkpoint yet, run make pretrain first")
    restore_checkpoint(supervised, model, optimizer)

    selfplay = make_manager(config.training)
    latest = selfplay.latest_step()
    iteration = 0 if latest is None else latest + 1
    save_checkpoint(selfplay, iteration, model, optimizer)
    print(f"supervised_{supervised.latest_step()} saved as self-play checkpoint {iteration}, make train continues from it")


if __name__ == "__main__":
    handoff(get_config())

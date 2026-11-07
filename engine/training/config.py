import copy

from ml_collections import config_dict

CONFIG = config_dict.ConfigDict({
    "training": {
        "learning_rate": 1e-3,  # size of each AdamW nudge
        "weight_decay": 1e-4,  # the W in AdamW: gently shrinks weights so they don't grow huge
        "batch_size": 256,  # positions per train step
        "value_weight": 1.0,  # how much the value loss counts vs the policy loss
        "games_per_iteration": 10,  # self-play games before each round of training
        "steps_per_iteration": 20,  # train steps after each round of games
        "buffer_size": 50_000,  # replay buffer keeps the most recent positions, oldest drop off
        "max_plies": 200,  # game cut off as a draw after this many half-moves
        "checkpoint_dir": "checkpoints",  # relative to the project root
        "checkpoint_every": 1,  # iterations between saves
    },
    "mcts": {
        "num_simulations": 50,  # searches per move (AlphaZero used 800)
        "c_puct": 1.5,  # how much the search explores moves it hasn't checked much
        "dirichlet_alpha": 0.3,  # root noise spikiness, lets self-play try moves the network ignores
        "dirichlet_epsilon": 0.25,  # share of root priors replaced by noise
        "temperature_moves": 30,  # moves picked randomly by visit share before switching to the best move
    },
    "supervised": {
        "num_workers": 8,  # processes encoding positions in parallel
        "chunk_positions": 200_000,  # X: positions read, encoded and trained on per round, ~1.15KB each so ~230MB
        "max_positions": 50_000_000,  # N: stop after training on this many positions, 0 = the whole file
        "val_positions": 50_000,  # first positions of the file, set aside to measure progress, never trained on
        "eval_every": 5_000_000,  # validation check + checkpoint every this many positions, keep it a multiple of chunk_positions
        "cp_scale": 400,  # value = tanh(centipawns / cp_scale), so +4 pawns ~ 0.76
    },
    "data": {
        "in_planes": 18,  # input maps: 8x8 planes per board from encoding/board.py
        "num_moves": 4672,  # action space size: 8x8 from-squares x 73 move types (see encoding/policy.py)
    },
    "model": {
        "num_layers": 3,  # number of conv layers
        "features": 64,  # output maps per conv layer
        "kernel_size": 3,  # 3 = 3x3 kernel
        "stride": 1,  # step between kernel positions; 1 keeps the board 8x8
        "padding": "SAME",  # keeps the board 8x8 through every layer
        "policy_head": {
            # in (8*8*features) and out (num_moves) are fixed; only the hidden layers between are set here
            "hidden": {
                "num": 0,  # number of hidden layers; 0 = straight from conv output to move scores
                "size": 512,  # outputs per hidden layer; unused while num is 0
            },
        },
        "value_head": {
            # in (8*8*features) and out (1, tanh) are fixed; only the hidden layers between are set here
            "hidden": {
                "num": 1,  # number of hidden layers
                "size": 64,  # outputs per hidden layer
            },
        },
    }
})



def get_config() -> config_dict.ConfigDict:
    config = copy.deepcopy(CONFIG)
    config.lock()  # values can change, but a typo'd key raises instead of being added
    return config
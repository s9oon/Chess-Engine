import copy

from ml_collections import config_dict

CONFIG = config_dict.ConfigDict({
    "training": {
        #TODO
    },
    "data": {
        # TODO: dataset path, eval scaling (cp/400)
        "in_planes": 18,  # input maps: 8x8 planes per board from the encoding
        "num_moves": 4672,  # action space size: 8x8 from-squares x 73 move types (see encoding.py)
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
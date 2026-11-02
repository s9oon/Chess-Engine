import copy

from ml_collections import config_dict

CONFIG = config_dict.ConfigDict({
    "data": {
        "in_planes": 18,  # input maps: 8x8 planes per board from the encoding
    },
    "model": {
        "num_layers": 3,  # number of conv layers
        "features": 64,  # output maps per conv layer
        "kernel_size": 3,  # 3 = 3x3 kernel
        "stride": 1,  # step between kernel positions; 1 keeps the board 8x8
        "padding": "SAME",  # keeps the board 8x8 through every layer
    }
})



def get_config() -> config_dict.ConfigDict:
    config = copy.deepcopy(CONFIG)
    config.lock()  # values can change, but a typo'd key raises instead of being added
    return config

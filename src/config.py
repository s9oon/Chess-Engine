import copy

from ml_collections import config_dict

CONFIG = config_dict.ConfigDict({
    #TODO
})



def get_config() -> config_dict.ConfigDict:
    config = copy.deepcopy(CONFIG)
    config.lock()  # values can change, but a typo'd key raises instead of being added
    return config

import os

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


class Config:
    conf = None

    def __init__(self, path: str):
        self.conf = OmegaConf.load(path)

    def get_config_value(self, key: str):
        try:
            return self.conf[key]
        except:
            raise KeyError(f"Cannot find the key {key} in config file")

    def __getitem__(self, key):
        return self.get_config_value(key)

    def keys(self):
        return self.conf.keys()

    def values(self):
        return self.conf.values()

    def items(self):
        return self.conf.items()

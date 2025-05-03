"""
Sacred experiment file
"""

# Sacred
from sacred import Experiment
from sacred.stflow import LogFileWriter
from sacred.observers import FileStorageObserver

# custom config hook
from utils.yaml_config_hook import yaml_config_hook


ex = Experiment("CPC")

# file output directory
ex.observers.append(FileStorageObserver("./experiments"))


@ex.config
def my_config():
    yaml_config_hook("./config/audio/config.yaml", ex)

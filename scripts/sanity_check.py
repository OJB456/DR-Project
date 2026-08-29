"""Run dataset/model forward-pass sanity checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import load_config
from src.training.train import run_sanity_check

if __name__ == "__main__":
    run_sanity_check(load_config("config.yaml"))

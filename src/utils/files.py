"""Utility functions for file operations"""

import json
from pathlib import Path

import yaml


def read_text_file(filename: str) -> list[str]:
    """Read txt file and return lines"""
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Optional: Remove leading/trailing whitespace
    return lines


def load_yaml(path: Path | str) -> dict:
    """Load yaml file to dict"""
    with open(path, "r") as file:
        yaml_dct = yaml.safe_load(file)
    return yaml_dct


def save_json(data: dict | list, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def save_yaml(dct: dict | list, path: Path | str) -> None:
    """Save dict as yaml file"""
    with open(path, "w") as file:
        yaml.dump(dct, file)


def save_txt_to_file(txt: str, filename: str):
    with open(filename, "w") as file:
        file.write(txt)

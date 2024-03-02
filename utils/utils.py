import yaml
import os


def load_yaml(file_path: str) -> dict:
    """
    Load YAML dict from a file.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(config: dict, file_path: str):
    """
    Save dict to a YAML file.
    """
    with open(file_path, "w") as f:
        yaml.dump(config, f)


def get_extensions_paths(directory: str, extension: str) -> list[str]:
    """Get paths of all files with '.{extension}' extension in the specified directory and its subdirectories."""
    gguf_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(f".{extension}")
    ]
    return gguf_paths


def get_filename(path: str) -> str:
    """Get the filename from the given path."""
    return os.path.basename(path)


def group_list(list, group_size):
    """Creates a list of lists of group_size"""
    k = len(list)
    grouped_list = []
    for k in range(0, k, group_size):
        grouped_list.append(list[k: k + group_size])
    return grouped_list

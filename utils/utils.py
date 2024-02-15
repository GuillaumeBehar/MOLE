import yaml


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

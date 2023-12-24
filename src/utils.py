import json
import tempfile
import yaml


def load_json(fp: str) -> dict | None:
    """
    This function loads a JSON file from the given path.

    Args:
        fp: The path to the JSON file.

    Returns:
        The contents of the JSON file as a dictionary.
    """

    try:
        with open(fp, "r") as f:
            json_file = json.load(f)
    except FileNotFoundError:
        json_file = None

    return json_file


def load_yaml(file_path: str) -> dict | None:
    """
    This function loads a YAML file from the given path.

    Args:
        file_path: The path to the YAML file.

    Returns:
        The contents of the YAML file as a dictionary.
    """

    try:
        with open(file_path, "r") as stream:
            yaml_file = yaml.safe_load(stream)
    except (yaml.YAMLError, FileNotFoundError):
        yaml_file = None

    return yaml_file


def load_text(file_path: str) -> str | None:
    """
    This function loads a text file from the given path.

    Args:
        file_path: The path to the text file.

    Returns:
        The contents of the text file as a string.
    """

    try:
        with open(file_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        text = None

    return text


def create_temp_file(api_data):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write the API data to the temporary file
        temp_file.write(api_data)

    # Return the path to the temporary file
    return temp_file.name
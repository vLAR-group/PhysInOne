import argparse
import yaml

def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)  # Use safe_load to avoid executing arbitrary code
    return data

def dict_to_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return argparse.Namespace(**d)

def read_yaml_to_namespce(file_path):
    d = read_yaml_to_dict(file_path)
    n = dict_to_namespace(d)
    return n

def convert_if_digit(value):
    """Convert the value to float or int if possible."""
    if isinstance(value, str):
        try:
            # Try converting to float first
            return float(value)
        except ValueError:
            try:
                # If float conversion fails, try converting to int
                return int(value)
            except ValueError:
                # If both conversions fail, return the original string
                return value
    return value  # Return the original value if it's not a string

def create_parser_from_namespace(ns):
    parser = argparse.ArgumentParser(description="recover parser from Namespcae")
    for key, value in vars(ns).items():
        parser.add_argument(f'--{key}', help=f'Value for {key}', default=value)
    return parser
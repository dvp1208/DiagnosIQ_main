import os
import re
import yaml
from dotenv import load_dotenv

load_dotenv()


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data


def replace_env_variables_in_config(config):
    for key, value in config.items():
        if isinstance(value, dict):
            replace_env_variables_in_config(value)
        elif isinstance(value, str):
            matches = re.findall(r'\$\{(\w+)\}', value)
            for match in matches:
                env_var_value = os.environ.get(match)
                if env_var_value:
                    config[key] = config[key].replace(
                        f'${{{match}}}', env_var_value)

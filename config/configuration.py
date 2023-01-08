"""
Module to define the configuration of the main program.

Classes:
    Config: Class to manipulate the configuration of the program.
"""

import configparser
import logging
import sys
import yaml

log = logging.getLogger(__name__)


def read_normalization_params(param_path):
    params = configparser.ConfigParser()
    params._interpolation = configparser.ExtendedInterpolation()
    params.read(param_path)
    return params


class GlobalConfig:

    """
    Class to manipulate the configuration of the program.
    """

    def __init__(self, config_path='config.yaml', param_path='normalization_parameters.ini'):
        self.data = None
        self.load(config_path)
        self.normalization_params = read_normalization_params(param_path)

    def get(self, key):
        """
        Fetch the configuration value of the specified key. If there are nested
        dictionaries, a dot notation can be used.
        """
        try:
            keys = key.split('.')
            value = self.data.copy()

            for v_key in keys:
                value = value[v_key]

            return value
        except KeyError:
            log.error(
                'Error get key '
                '{}'.format(
                    key
                )
            )
            sys.exit(1)

    def load(self, config_path='config.yaml'):
        """
        Loads configuration from configuration YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                try:
                    self.data = yaml.full_load(f)
                except yaml.YAMLError as e:
                    log.error(
                        'Error parsing yaml of configuration file '
                        '{}'.format(
                            e
                        )
                    )
                    sys.exit(1)
        except FileNotFoundError:
            log.error(
                'Error opening configuration file {}'.format(self.config_path)
            )
            sys.exit(1)

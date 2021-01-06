import argparse

from core.utils.parser_utils import add_cg_arguments, munch_cg_params

def get_parser():
    """Parse command line arguments and return args namespace."""

    parser = argparse.ArgumentParser(description='Plotting a hybrid control barrier function')

    parser.add_argument('')
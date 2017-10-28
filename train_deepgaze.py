import os
import argparse
import numpy as np
from ops.tf_model_conv import train_and_eval
from ops.utils import get_dt
import sys
from config import deepgazeConfig


def main(show_output):
    config = deepgazeConfig()

    if show_output is not None:
        config.show_output = True

    # Trains a random forest model
    train_and_eval(config)
    dt_stamp = get_dt()  # date-time stamp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--show_output', dest='show_output',
        default=None, help='Show example output every 20 steps (for debugging purposes).')
    args=parser.parse_args()
    main(**vars(args))

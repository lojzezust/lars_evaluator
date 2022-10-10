from PIL import Image
import numpy as np
import argparse

from evaluator import Evaluator
from evaluator.config import get_cfg


def main():
    parser = argparse.ArgumentParser(description='LaRS evaluation script')

    parser.add_argument('config', help='Configuration file', type=str)
    parser.add_argument('method', help='Method to evaluate. Prediction dir should contain a directory with the same name, containing the predicted segmentation masks',
                        type=str)


    args = parser.parse_args()

    cfg = get_cfg(args.config)

    evaluator = Evaluator(cfg)
    results = evaluator.evaluate(args.method)


if __name__=='__main__':
    main()

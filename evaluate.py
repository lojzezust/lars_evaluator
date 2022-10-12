from PIL import Image
import numpy as np
import argparse

from evaluator import Evaluator
from evaluator.config import get_cfg
from multiprocessing import Pool
from evaluator.utils import TqdmPool
from tqdm.auto import tqdm

WORKERS=8

def main():
    parser = argparse.ArgumentParser(description='LaRS evaluation script')

    parser.add_argument('config', help='Configuration file', type=str)
    parser.add_argument('methods', nargs='+', help='Method(s) to evaluate. Prediction dir should contain a directory with the same name, containing the predicted segmentation masks',
                        type=str)

    parser.add_argument('--workers', default=WORKERS, type=int)


    args = parser.parse_args()

    cfg = get_cfg(args.config)

    evaluator = Evaluator(cfg)

    if len(args.methods) > 1:
        with TqdmPool(WORKERS) as pool:
            list(tqdm(pool.imap_unordered(evaluator.evaluate, args.methods), total=len(args.methods)))
    else:
        results = evaluator.evaluate(args.methods[0])
        print(results)


if __name__=='__main__':
    main()

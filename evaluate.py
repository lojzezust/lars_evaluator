from PIL import Image
import numpy as np
import argparse
from multiprocessing import Pool
from tqdm.auto import tqdm
import os.path as osp

from lars_eval import SemanticEvaluator, PanopticEvaluator
from lars_eval.config import get_cfg
from lars_eval.utils import TqdmPool

WORKERS=8

class MethodEvaluator():
    def __init__(self, cfg, evaluator):
        self.cfg = cfg
        self.evaluator = evaluator

    def evaluate_method(self, method):
        pred_dir = osp.join(self.cfg.PATHS.PREDICTIONS, method)
        output_dir = osp.join(self.cfg.PATHS.RESULTS, method)
        return self.evaluator.evaluate(pred_dir, output_dir, display_name=method)


def main():
    parser = argparse.ArgumentParser(description='LaRS evaluation script')

    parser.add_argument('config', help='Configuration file', type=str)
    parser.add_argument('methods', nargs='+', help='Method(s) to evaluate. Prediction dir should contain a directory with the same name, containing the predicted segmentation masks',
                        type=str)

    parser.add_argument('--workers', default=WORKERS, type=int)


    args = parser.parse_args()

    cfg = get_cfg(args.config)

    if cfg.MODE == 'semantic':
        evaluator = SemanticEvaluator(cfg)
    elif cfg.MODE == 'panoptic':
        evaluator = PanopticEvaluator(cfg)
    else:
        raise ValueError('Unknown mode: %s' % cfg.MODE)

    my_evaluator = MethodEvaluator(cfg, evaluator)

    if len(args.methods) > 1:
        with TqdmPool(WORKERS) as pool:
            list(tqdm(pool.imap_unordered(my_evaluator.evaluate_method, args.methods), total=len(args.methods)))
    else:
        results = my_evaluator.evaluate_method(args.methods[0])
        print(results)


if __name__=='__main__':
    main()

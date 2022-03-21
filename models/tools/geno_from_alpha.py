import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.append(os.getcwd())

from models.yolo_search import Model, parse_model

logger = logging.getLogger(__name__)

def main(hyp, opt, mode='from_alphas'):
    # construct model
    if mode == 'from_alphas':
      model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')) # create

      # load alphas
      alphas_file = os.path.join(opt.save_dir, 'exp/alphas/49.yaml')
      with open(alphas_file) as f:
          alphas = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
      for key in alphas.keys():
          alphas[key] = torch.tensor(alphas[key])
      model.load_state_dict(alphas, strict=False)
    else:
      # load weights
      state_dict = torch.load(opt.weights)
      model = state_dict['model']
#      model.edge_arch_parameters = []
    geno, model_yaml = model.genotype()
    del model_yaml['geno']
    # save genotype
    geno_file = os.path.join(opt.save_dir, 'exp/genotypes/49.yaml')
    with open(geno_file, encoding='utf-8', mode='w') as f:
      try:
          yaml.dump(data=model_yaml, stream=f, allow_unicode=True)
      except Exception as e:
          print(e)
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--weights', type=str, default='', help='hyperparameters path')
    parser.add_argument('--save_dir', type=str, default='runs/train', help='hyperparameters path')

    opt = parser.parse_args()
    
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    if opt.weights is not None:
      main(hyp, opt, mode='from_weights')
    else:
      main(hyp, opt, mode='from_alphas')



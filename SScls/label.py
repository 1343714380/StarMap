from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
import datetime
from utils.logger import Logger
import torch
import torch.utils.data
from opts import opts
import ref
from model import getModel, saveModel
opt = opts().parse()

from myPascal3D import Pascal3D as Dataset, get_dataloader
from trainCls import label_train_set, label_val_set

def main():
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))
  model, optimizer = getModel(opt)
  
  if opt.GPU > -1:
    print('Using GPU', opt.GPU)
    model = model.cuda(opt.GPU)
  
  #val_loader = get_dataloader(opt,'val')
  train_loader = get_dataloader(opt,'train')


  label_train_set(opt, train_loader, model)
  #label_val_set(opt, val_loader, model)

  logger.close()

if __name__ == '__main__':
  main()

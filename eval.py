#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""
import os
import gin
from gin.config import _CONFIG
import torch
import pickle
import logging
from functools import partial
logger = logging.getLogger(__name__)

from src import dataset
from src import callbacks as avail_callbacks 
from src.model import MMTM_MVCNN
from src.training_loop import evalution_loop
from src.utils import gin_wrap

from train import blend_loss, acc

@gin.configurable
def eval_(save_path, 
          target_data_split,
          pretrained_weights_path,
          batch_size=128, 
          callbacks=[],
          ):

    model = MMTM_MVCNN()
    train, val, testing = dataset.get_mvdcndata(batch_size=batch_size)

    if target_data_split == 'test':
        target_data = testing
    elif target_data_split == 'train':
        target_data = train
    elif target_data_split == 'val':
        target_data = val
    else:
        raise NotImplementedError 

    # Create dynamically callbacks
    callbacks_constructed = []
    for name in callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks_constructed.append(clbk)

    evalution_loop(model=model, 
                   loss_function=blend_loss, 
                   metrics=[acc],
                   config=_CONFIG, 
                   save_path=save_path, 
                   test=target_data,  
                   test_steps=len(target_data),
                   custom_callbacks=callbacks_constructed,
                   pretrained_weights_path=pretrained_weights_path)


if __name__ == "__main__":
    gin_wrap(eval_)

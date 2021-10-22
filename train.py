#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: train.py save_to_folder configs/cnn.gin.
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
from src.training_loop import training_loop
from src.utils import gin_wrap


def blend_loss(y_hat, y):
    loss_func = torch.nn.CrossEntropyLoss()
    losses = []
    for y_pred in y_hat:
        losses.append(loss_func(y_pred, y))

    return sum(losses)


def acc(y_pred, y_true):
    if isinstance(y_pred, list): 
        y_pred = torch.mean(torch.stack([out.data for out in y_pred], 0), 0)
    _, y_pred = y_pred.max(1)
    if len(y_true)==2:
        acc_pred = (y_pred == y_true[0]).float().mean()
    else:
        acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100


@gin.configurable
def train(save_path, wd, lr, momentum, batch_size, callbacks=[]):
    model = MMTM_MVCNN()
    train, valid, test = dataset.get_mvdcndata(batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), 
        lr=lr, 
        weight_decay=wd, 
        momentum=momentum)

    callbacks_constructed = []
    for name in callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks_constructed.append(clbk)

    training_loop(model=model, 
        optimizer=optimizer, 
        loss_function=blend_loss, 
        metrics=[acc],
        train=train, valid=valid, test=test, 
        steps_per_epoch=len(train),
        validation_steps=len(valid),
        test_steps=len(test),
        save_path=save_path, 
        config=_CONFIG,
        custom_callbacks=callbacks_constructed
    )


if __name__ == "__main__":
    gin_wrap(train)

# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
import tqdm
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
import gin

from src.callbacks import ModelCheckpoint, LambdaCallback
from src.utils import save_weights
from src.framework import Model_

logger = logging.getLogger(__name__)

types_of_instance_to_save_in_csv = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.float128, str)
types_of_instance_to_save_in_history = (int, float, complex, np.int64, np.int32, np.float32, np.float64, np.ndarray, np.float128,str)

def _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor, save_with_structure=False):
    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H)))

    callbacks.append(
        LambdaCallback(
            on_epoch_end=partial(_save_history_csv, 
            save_path=save_path, 
            H=H, 
            save_with_structure=save_with_structure)
        )
    )
    
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
                                 save_best_only=True,
                                 mode='max',
                                 filepath=os.path.join(save_path, "model_best_val.pt")))
    
    def save_weights_fnc(epoch, logs):
        logger.info("Saving model from epoch " + str(epoch))
        save_weights(model, optimizer, os.path.join(save_path, "model_last_epoch.pt"))

    callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

    return callbacks


def _save_history_csv(epoch, logs, save_path, H, save_with_structure = False):
    out = ""
    for key, value in logs.items():
        if isinstance(value, types_of_instance_to_save_in_csv):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    logger.info("Saving history to " + os.path.join(save_path, "history.csv"))
    H_tosave = {}
    for key, value in H.items():
        if isinstance(value[-1], types_of_instance_to_save_in_csv):
            H_tosave[key] = value
    pd.DataFrame(H_tosave).to_csv(os.path.join(save_path, "history.csv"), index=False)
    if save_with_structure: 
        with open(os.path.join(save_path, "history.pickle"), 'wb') as f:
            pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)


def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if key not in H:
            H[key] = [value]
        else:
            H[key].append(value)


def _load_pretrained_model(model, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model']) 
    model.load_state_dict(model_dict, strict=False)
    logger.info("Done reloading!")


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, config, 
                  save_path,  steps_per_epoch, 
                  train=None, valid=None, test=None,
                  test_steps=None, validation_steps=None,
                  use_gpu = False, device_numbers = [0],
                  custom_callbacks=[], 
                  checkpoint_monitor="val_acc", 
                  n_epochs=100, 
                  verbose=True, 
                  nummodalities=2):

    callbacks = list(custom_callbacks)

    history_csv_path = os.path.join(save_path, "history.csv")
    history_pkl_path = os.path.join(save_path, "history.pkl")

    logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
    os.system("rm " + history_pkl_path)
    os.system("rm " + history_csv_path)

    H = {}

    callbacks += _construct_default_callbacks(model, optimizer, H, 
        save_path, checkpoint_monitor, custom_callbacks)
    
    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_optimizer(optimizer)
        clbk.set_config(config)

    model = Model_(model=model, 
        optimizer=optimizer, 
        loss_function=loss_function, 
        metrics=metrics,
        verbose=verbose,
        nummodalities=nummodalities,
        )
            
    for clbk in callbacks:
        clbk.set_model_pytoune(model)

    if use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(device_numbers[0]))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
        
    _ = model.train_loop(train,
                            valid_generator=valid,
                            test_generator=test,
                            test_steps=test_steps,
                            validation_steps=validation_steps,
                            steps_per_epoch=steps_per_epoch,
                            epochs=n_epochs - 1, 
                            callbacks=callbacks,
                            )

def _construct_default_eval_callbacks(H, save_path, save_with_structure):
    
    history_batch = os.path.join(save_path, 'eval_history_batch')
    if not os.path.exists(history_batch):
        os.mkdir(history_batch)

    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H)))

    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, 
                                                        save_path=history_batch, 
                                                        H=H, 
                                                        save_with_structure=save_with_structure)))

    return callbacks

@gin.configurable
def evalution_loop(model, loss_function, metrics, config, 
                   save_path,
                   test=None,  test_steps=None,
                   use_gpu = False, device_numbers = [0], 
                   custom_callbacks=[],  
                   pretrained_weights_path=None,
                   save_with_structure=False,
                   nummodalities=2,
                  ):

    
    _load_pretrained_model(model, pretrained_weights_path)

    history_csv_path = os.path.join(save_path, "eval_history.csv")
    history_pkl_path = os.path.join(save_path, "eval_history.pkl")

    logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
    os.system("rm " + history_pkl_path)
    os.system("rm " + history_csv_path)

    H = {}
    callbacks = list(custom_callbacks)
    callbacks += _construct_default_eval_callbacks(
        H, 
        save_path, 
        save_with_structure
    )

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_config(config)

    model = Model_(model=model, 
                   optimizer=None, 
                   loss_function=loss_function, 
                   metrics=metrics,
                   nummodalities=nummodalities)

    if use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(device_numbers[0]))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
    
    model.eval_loop(
        test,  
        epochs=0,
        test_steps=test_steps,
        callbacks=callbacks
    )



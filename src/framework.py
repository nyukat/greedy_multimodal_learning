import warnings
import numpy as np
import timeit
import torch
from functools import partial
import math
import itertools

from src.callbacks import (
     ValidationProgressionCallback, 
     ProgressionCallback,
     CallbackList, 
     Callback
)
from src.utils import numpy_to_torch, torch_to_numpy, torch_to

import logging
logger = logging.getLogger(__name__)

warning_settings = {
    'batch_size': 'warn'
}

def cycle(iterable): 
    while True:
        for x in iterable:
            yield x


def _get_step_iterator(steps, generator):
    count_iterator = range(1, steps + 1) if steps is not None else itertools.count(1)
    generator = cycle(generator) if steps is not None else generator
    return zip(count_iterator, generator)


class StepIterator:
    def __init__(self, generator, steps_per_epoch, callback, metrics_names, nummodalities):
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self.callback = callback
        self.metrics_names = metrics_names
        self.nummodalities = nummodalities

        self.losses_sum = 0.
        self.metrics_sum = np.zeros(len(self.metrics_names))
        self.metrics_permodal_sum = np.zeros((nummodalities, len(self.metrics_names)))
        self.sizes_sum = 0.
        self.extra_lists = {}
        self.indices_list = []

        self.defaultfields = ['indices',
                              'loss', 
                              'metrics', 
                              'viewwises_metrics', 
                              'number',
                              'size'
                              ]

    @property
    def loss(self):
        if self.sizes_sum==0:
            return 0
        else:
            return self.losses_sum / self.sizes_sum

    @property
    def metrics(self):
        if self.sizes_sum==0:
            return dict(zip(self.metrics_names, np.zeros(len(self.metrics_names))))
        else:
            metrics_dict = dict(zip(self.metrics_names, self.metrics_sum / self.sizes_sum))
            for i in range(self.nummodalities):
                names = [f'{x}_modal_{i}' for x in self.metrics_names]
                metrics_dict.update(dict(zip(names, self.metrics_permodal_sum[i]/self.sizes_sum)))

            return metrics_dict

    @property
    def indices(self):
        if self.sizes_sum==0:
            return []
        elif self.indices_list[0] is None:
            return []
        else:
            return np.concatenate(self.indices_list, axis=0)

    def __iter__(self):
        for batch_ind, data in _get_step_iterator(self.steps_per_epoch, self.generator):
            batch_begin_time = timeit.default_timer()
            self.callback.on_batch_begin(batch_ind, {})
            self.callback.on_forward_begin(batch_ind, data) 

            step_data = {'number': batch_ind}
            step_data['indices'] = data[0]
            yield step_data, data[1:]

            self.losses_sum += step_data['loss'] * step_data['size']
            self.metrics_sum += step_data['metrics'] * step_data['size']
            self.metrics_permodal_sum += step_data['viewwises_metrics'] * step_data['size']
            self.sizes_sum += step_data['size']
            self.indices_list.append(step_data['indices'])

            metrics_dict = dict(zip(self.metrics_names, step_data['metrics']))

            for i in range(self.nummodalities):
                names = [f'{x}_modal_{i}' for x in self.metrics_names]
                metrics_dict.update(dict(zip(names, step_data['viewwises_metrics'][i])))

            for key, value in step_data.items():
                if key not in self.defaultfields:
                    if key in self.extra_lists:
                        self.extra_lists[key].append(value)
                    else:
                        self.extra_lists[key] = [value]
                    
            batch_total_time = timeit.default_timer() - batch_begin_time

            batch_logs = {'batch': batch_ind, 'size': step_data['size'], 
                          'time': batch_total_time, 'batch_begin_time': batch_begin_time, 
                          'loss': step_data['loss'], **metrics_dict}

            self.callback.on_batch_end(batch_ind, batch_logs)


class Model_:
    def __init__(self, model, optimizer, loss_function, nummodalities, *, metrics=[], 
        verbose=True, hyper_optim=None, vg=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.metrics_names = [metric.__name__ for metric in self.metrics]
        self.device = None
        self.verbose = verbose
        self.verbose_logs = {} 
        self.nummodalities = nummodalities
        self.curation_mode=False
        self.caring_modality=None

    def _compute_loss_and_metrics(self, x, y):
        x, y = self._process_input(x, y)
        x = x if isinstance(x, (list, tuple)) else (x, )

        self.minibatch_data = (x, y)

        pred_y_eval, pred_y, scales, squeezed_mps  = self.model(*x, 
            curation_mode=self.curation_mode, 
            caring_modality=self.caring_modality)

        loss = self.loss_function(pred_y, y)

        record = {} 

        with torch.no_grad():
            record['metrics'] = self._compute_metrics(pred_y_eval, y)
            record['viewwises_metrics'] = self._compute_metrics_multiple_inputs(pred_y, y)

        if self.model.saving_mmtm_scales:
            record['mmtmscales_list'] = scales
        if self.model.saving_mmtm_squeeze_array:
            record['squeezedmaps_array_list'] = squeezed_mps

        return loss, record

    def _process_input(self, *args):
        args = numpy_to_torch(args)
        if self.device is not None:
            args = torch_to(args, self.device)
        return args[0] if len(args) == 1 else args

    def _compute_metrics(self, pred_y, y):
        return np.array([float(metric(pred_y, y)) for metric in self.metrics])

    def _compute_metrics_multiple_inputs(self, list_pred_y, y):
        return np.array([self._compute_metrics(pred_y, y) for pred_y in list_pred_y])

    def _get_batch_size(self, x, y):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            return len(x)
        if torch.is_tensor(y) or isinstance(y, np.ndarray):
            return len(y)
        if warning_settings['batch_size'] == 'warn':
            warnings.warn("When 'x' or 'y' are not tensors nor Numpy arrays, "
                          "the batch size is set to 1 and, thus, the computed "
                          "loss and metrics at the end of each epoch is the "
                          "mean of the batches' losses and metrics. To disable "
                          "this warning, set\n"
                          "from poutyne.framework import warning_settings\n"
                          "warning_settings['batch_size'] = 'ignore'")
        return 1

    def _transfer_optimizer_state_to_right_device(self):
        # Since the optimizer state is loaded on CPU, it will crashed when the
        # optimizer will receive gradient for parameters not on CPU. Thus, for
        # each parameter, we transfer its state in the optimizer on the same
        # device as the parameter itself just before starting the optimization.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    for _, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v) and p.device != v.device:
                            v.data = v.data.to(p.device)

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        if isinstance(self.loss_function, torch.nn.Module):
            self.loss_function.to(self.device)

        for metric in self.metrics:
            if isinstance(metric, torch.nn.Module):
                metric.to(self.device)

        return self

    def _eval_generator(self, generator, phase, *, steps=None):
        if steps is None:
            steps = len(generator)
        
        step_iterator = StepIterator(
                            generator, 
                            steps, 
                            ValidationProgressionCallback(
                                phase=phase, 
                                steps=steps, 
                                metrics_names=['loss'] + self.metrics_names
                            ), 
                            self.metrics_names,
                            self.nummodalities
                        )

        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in step_iterator:
                step['size'] = self._get_batch_size(x, y)
                loss_tensor, info = self._compute_loss_and_metrics(x, y)
                step['loss'] = float(loss_tensor)
                step.update(info)

        metrics_dict = {
            f'{phase}_{metric_name}' : metric for metric_name, metric in step_iterator.metrics.items()
        }

        info_dict = {f'{phase}_loss' : step_iterator.loss, 
            f'{phase}_indices': step_iterator.indices,
            **{f'{phase}_{k}':v for k, v in step_iterator.extra_lists.items()},
            **metrics_dict
        }

        return info_dict

    def eval_loop(self, test_generator, *,  test_steps=None, epochs=1, callbacks=[]):
        callback_list = CallbackList(callbacks)
        callback_list.set_model_pytoune(self)
        callback_list.on_train_begin({})
        epoch = 0
        while epoch <=epochs:
            epoch_begin_time = timeit.default_timer()
            callback_list.on_epoch_begin(epoch, {})
            test_dict = self._eval_generator(test_generator, 'test', steps=test_steps)

            test_dict['epoch'] = epoch
            test_dict['time'] = timeit.default_timer() - epoch_begin_time
            test_dict['epoch_begin_time'] = epoch_begin_time

            callback_list.on_epoch_end(epoch, test_dict)
            
            epoch+=1

    def train_loop(self,
                      train_generator,
                      test_generator=None,
                      valid_generator=None,
                      *,
                      epochs=1000,
                      steps_per_epoch=None,
                      validation_steps=None,
                      test_steps=None,
                      callbacks=[],
                      ):
        
        self._transfer_optimizer_state_to_right_device()

        callback_list = CallbackList(callbacks)
        callback_list.append(ProgressionCallback())
        callback_list.set_model_pytoune(self)
        callback_list.set_params({'epochs': epochs, 'steps': steps_per_epoch})

        self.stop_training = False

        callback_list.on_train_begin({})
        val_dict, test_dict = {}, {}
        for epoch in range(1, epochs+1):
            callback_list.on_epoch_begin(epoch, {})
            
            epoch_begin_time = timeit.default_timer()

            # training
            train_step_iterator = StepIterator(train_generator,
                                               steps_per_epoch,
                                               callback_list,
                                               self.metrics_names,
                                               self.nummodalities
                                               )
            self.model.train(True)
            with torch.enable_grad():
                for step, (x, y) in train_step_iterator: 
                    step['size'] = self._get_batch_size(x, y)

                    self.optimizer.zero_grad()
                    loss_tensor, info = self._compute_loss_and_metrics(x, y)

                    loss_tensor.backward()
                    callback_list.on_backward_end(step['number'])
                    self.optimizer.step()  

                    loss = loss_tensor.item()
                    step.update(info)
                    step['loss'] = loss
                    
                    if math.isnan(step['loss']): 
                        self.stop_training = True

            train_dict = {'loss': train_step_iterator.loss, 
                    'train_indices': train_step_iterator.indices,
                    **{f'train_{k}':v for k, v in train_step_iterator.extra_lists.items()},
                    **train_step_iterator.metrics}
            
            # validation
            val_dict = self._eval_generator(valid_generator, 'val', steps=validation_steps)
            # test
            test_dict = self._eval_generator(test_generator, 'test', steps=test_steps)
           
            epoch_log = {
                'epoch': epoch, 
                'time': timeit.default_timer() - epoch_begin_time, 
                'epoch_begin_time': epoch_begin_time,
                **train_dict, **val_dict, **test_dict
            }
             
            callback_list.on_epoch_end(epoch, epoch_log)

            if self.stop_training: break

        callback_list.on_train_end({})

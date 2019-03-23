# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""Callback functions that can be used to track various status during epoch."""
from __future__ import absolute_import

import logging
import math
import time
from .model import save_checkpoint

def module_checkpoint(mod, prefix, period=1, save_optimizer_states=False):
    """Callback to checkpoint Module to prefix every epoch.

    Parameters
    ----------
    mod : subclass of BaseModule
        The module to checkpoint.
    prefix : str
        The file prefix for this checkpoint.
    period : int
        How many epochs to wait before checkpointing. Defaults to 1.
    save_optimizer_states : bool
        Indicates whether or not to save optimizer states for continued training.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))
    # pylint: disable=unused-argument
    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            mod.save_checkpoint(prefix, iter_no + 1, save_optimizer_states)
    return _callback


def do_checkpoint(prefix, period=1):
    """A callback that saves a model checkpoint every few epochs.
    Each checkpoint is made up of a couple of binary files: a model description file and a
    parameters (weights and biases) file. The model description file is named
    `prefix`--symbol.json and the parameters file is named `prefix`-`epoch_number`.params

    Parameters
    ----------
    prefix : str
        Prefix for the checkpoint filenames.
    period : int, optional
        Interval (number of epochs) between checkpoints. Default `period` is 1.

    Returns
    -------
    callback : function
        A callback function that can be passed as `epoch_end_callback` to fit.

    Example
    -------
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... epoch_end_callback  = mx.callback.do_checkpoint("mymodel", 1))
    Start training with [cpu(0)]
    Epoch[0] Resetting Data Iterator
    Epoch[0] Time cost=0.100
    Saved checkpoint to "mymodel-0001.params"
    Epoch[1] Resetting Data Iterator
    Epoch[1] Time cost=0.060
    Saved checkpoint to "mymodel-0002.params"
    """
    period = int(max(1, period))
    def _callback(iter_no, sym, arg, aux):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback


def log_train_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0 and param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                             param.epoch, param.nbatch, name, value)
            if auto_reset:
                param.eval_metric.reset()
    return _callback


class Speedometer(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    Epoch[0] Batch [10] Speed: 1910.41 samples/sec  Train-accuracy=0.200000
    Epoch[0] Batch [20] Speed: 1764.83 samples/sec  Train-accuracy=0.400000
    Epoch[0] Batch [30] Speed: 1740.59 samples/sec  Train-accuracy=0.500000
    """
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f'*len(name_value)
                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class CSVSpeedometer(object):
    """
    Log 
      - Training Throughput (Samples per Second), 
      - Memory Usage (from `nvidia-smi`)
      - Evaluation Metrics 
      - Power and Energy Consumption
    periodically, and at the same time, dump the results using CSV file.
    :param csv_fname: Output CSV Filename
    :param batch_size: Training Batch Size, used for computing Throughput
    :param frequency : How frequent will those aforementioned Metrics be Logged
    :param auto_reset: Whether to reset the Evaluation Metrics after each Log
    """
    def __init__(self, batch_size, frequent=50, auto_reset=True, 
                 csv_fname='/tmp/mxnet_speedometer.csv'):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        # `global_step` records the number of training batches.
        # It is incremented every time the `Speedometer` is called.
        self.global_step = 0
        
        import subprocess
        sp = subprocess.Popen(['nvidia-smi', 
                               '--query-gpu=power.draw',
                               '--format=csv,noheader,nounits'],
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        query_result = sp.communicate()[0].decode("utf-8").rstrip().split("\n")
        # initial energy to an array of zeros
        self.energy = [0.0 for _ in query_result]

        import os
        try:
            # cleanup previous output file, if there exists
            os.remove(csv_fname)
        except OSError:
            pass
        self.csv_fname = csv_fname

    def __call__(self, param):
        """
        Callback to show Training Throughput, Memory Usage, Evaluation Metrics
        """
        count = param.nbatch

        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                # Training Throughput
                time_diff = time.time() - self.tic
                speed = self.frequent * self.batch_size / time_diff

                import subprocess

                training_log_entry = ['global_step', '%d' % self.global_step,
                                      'wc_time',     '%f' % time.time(),
                                      'throughput',  '%f' % speed]

                # NVIDIA Docker containers have trouble in querying process names.
                # Therefore process names are temporarily ignored.
                sp = subprocess.Popen(['nvidia-smi', 
                                       '--query-compute-apps=pid,used_gpu_memory', 
                                       '--format=csv,noheader,nounits'],
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
                query_result = sp.communicate()[0].decode("utf-8").rstrip().split("\n")
                memory_usage = []

                if len(query_result) > 1:
                    logging.info("There are more than 1 compute application running on the GPU.")

                for line in query_result:
                    pid, used_gpu_memory = line.split(", ")
                    pid, used_gpu_memory = int(pid), int(used_gpu_memory)

                    memory_usage_tag = "memory_usage-pid_%d" % pid
                    memory_usage.append((memory_usage_tag, used_gpu_memory))
                    training_log_entry.extend([memory_usage_tag, 
                                               '%d'%used_gpu_memory])
                
                sp = subprocess.Popen(['nvidia-smi', 
                                       '--query-gpu=power.draw',
                                       '--format=csv,noheader,nounits'],
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
                query_result = sp.communicate()[0].decode("utf-8").rstrip().split("\n")
                pe_usage = []
                
                for i, line in enumerate(query_result):
                    power = float(line)
                    self.energy[i] += power * time_diff

                    pe_usage_tag = "pe_usage-dev_%d" % i
                    pe_usage.append((pe_usage_tag, power, self.energy[i]))
                    training_log_entry.extend(['power-dev_%d' %i, '%.2f'%power,
                                               'energy-dev_%d'%i, '%.2f'%self.energy[i]])

                # Evaluation Metrics
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()

                    for name, value in dict(name_value).items():
                        training_log_entry.extend(['eval_metric-%s' % name, '%f' % value])

                    msg  = 'Global Step[%d] Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f' * len(name_value)
                    msg += '\tMemory Usage (MB): '
                    msg += '\t%s=%d' * len(memory_usage)
                    msg += '\tPE Usage (W, J): '
                    msg += '\t%s=%.2f,%.2f' * len(pe_usage)
                    logging.info(msg, self.global_step, param.epoch, count, speed,
                                 *sum(name_value + memory_usage + pe_usage, ()))

                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)

                with open(self.csv_fname, 'a') as fout:
                    fout.write(",".join(training_log_entry))
                    fout.write('\n')

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

        self.global_step += 1 # increase `global_Step` by 1


class ProgressBar(object):
    """Displays a progress bar, indicating the percentage of batches processed within each epoch.

    Parameters
    ----------
    total: int
        total number of batches per epoch
    length: int
        number of chars to define maximum length of progress bar

    Examples
    --------
    >>> progress_bar = mx.callback.ProgressBar(total=2)
    >>> mod.fit(data, num_epoch=5, batch_end_callback=progress_bar)
    [========--------] 50.0%
    [================] 100.0%
    """
    def __init__(self, total, length=80):
        self.bar_len = length
        self.total = total

    def __call__(self, param):
        """Callback to Show progress bar."""
        count = param.nbatch
        filled_len = int(round(self.bar_len * count / float(self.total)))
        percents = math.ceil(100.0 * count / float(self.total))
        prog_bar = '=' * filled_len + '-' * (self.bar_len - filled_len)
        logging.info('[%s] %s%s\r', prog_bar, percents, '%')


class LogValidationMetricsCallback(object):
    """Just logs the eval metrics at the end of an epoch."""

    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)

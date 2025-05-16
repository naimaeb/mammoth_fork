import os
import sys
sys.path.append(os.getcwd())

from string import Template
import socket
import time
import os
from gridsearcher import GridSearcher
from tqdm import tqdm
import argparse
import psutil
import math

def main(gpus, dist=False, max_jobs=1, param_dict=None):
    state_finished = '/nfs/scistore19/alistgrp/imodoran/workplace/BlockOpt2GeomRep/mammoth/data/state.finished'
    if os.path.isfile(state_finished):
        os.remove(state_finished)

    gs = GridSearcher(
        script='/nfs/scistore19/alistgrp/imodoran/workplace/BlockOpt2GeomRep/mammoth/utils/main.py',
        defaults=dict())

    gs.add_param('model', 'er')
    gs.add_param('buffer_size', 200)
    gs.add_param('non_verbose', 1)
    gs.add_param('num_workers', 0)

    # gs.add_param('fitting_mode', 'early_stopping')
    # gs.add_param('early_stopping_patience', '0')
    # gs.add_param('early_stopping_freq', '1')
    # gs.add_param('early_stopping_metric', 'loss')
    # gs.add_param('early_stopping_epsilon', '1e-3')
    gs.add_param('enable_other_metrics', '1')
    gs.add_param('dataset_config', 'dataset-bogr')
    gs.add_param('wandb_entity', 'ist')
    gs.add_param('wandb_project', 'ionut_mammoth')
    gs.add_param('optimizer', 'blockadam')
    gs.add_param('lr', '1e-3')

    gs.run(
        torchrun=False,
        launch_blocking=0,
        scheduling=dict(
            distributed_training=dist,
            max_jobs_per_gpu=max_jobs,
            gpus=gpus,
            params_values=param_dict
        ),
        param_name_for_exp_root_folder='base_path',
        exp_folder=os.path.join('/nfs/scistore19/alistgrp/imodoran/workplace/BlockOpt2GeomRep/mammoth/data'))

if __name__ == '__main__':
    gpus = {
	'gpu268': [1, 2, 4, 5],
        'gpu270': [0, 1, 2, 3, 4, 5, 6, 7],
        'gpu272': [0, 1, 2, 3, 4, 5, 6, 7],
        'gpu273': [0, 1, 2, 3, 4, 5, 6, 7],
        'gpu274': [0, 1, 2, 3, 4, 5, 6, 7],
    }

    main(
        max_jobs=1,
        dist=False,
        # dist=True,

        gpus=gpus[socket.gethostname()],

        param_dict={
            # 'dataset': ['perm-mnist'],
            'dataset': ['seq-cifar10'],

            # 'seed': [42],
            'seed': [7, 42, 1234],
            # 'seed': [2-1, 4-1, 8-1, 16-1, 32-1, 64-1, 128-1, 256-1, 512-1, 1024-1, 4096-1, 8192-1, 16384-1, 32768-1],

            ##### SEQ-CIFAR10
            'model': ['der'], 'alpha': ['0.3'],
            # 'model': ['derpp'], 'alpha': ['0.1'], 'beta': ['0.5'],

            'optimizer': ['sgd'], 'lr': ['0.03'],

            'optimizer': ['shmp'], 'lr': ['0.03'],
            # 'optimizer': ['muon-ns'], 'lr': ['5e-5', '3e-5', '1e-5', '5e-4', '3e-4', '1e-4', '5e-3', '3e-3', '1e-3'],
            # 'optimizer': ['blockadam'], 'lr': ['5e-5', '3e-5', '1e-5'], 'blockadam_block_size': [100], 'blockadam_func': ['ns'],
            # 'optimizer': ['blockadam'], 'lr': ['5e-5', '3e-5', '1e-5'], 'blockadam_block_size': [32], 'blockadam_func': ['abs'],
            # 'optimizer': ['mfac'], 'lr': ['1e-3', '1e-4', '1e-5'], 'blockadam_block_size': [32], 'blockadam_func': ['abs'],



            ##### PERM-MNIST
            # 'model': ['der'], 'alpha': ['0.5', '1'],
            # # 'model': ['derpp'], 'alpha': ['0.5', '1'], 'beta': ['0.5', '1'],
            # 'optimizer': ['sgd'], 'lr': ['0.1', '0.2'],
            # # 'optimizer': ['shmp'], 'lr': ['0.1', '0.2'],
            # # 'optimizer': ['muon-ns'], 'lr': ['5e-5', '3e-5', '1e-5', '5e-4', '3e-4', '1e-4', '5e-3', '3e-3', '1e-3'],
            # # 'optimizer': ['blockadam'], 'lr': ['5e-5', '3e-5', '1e-5'], 'blockadam_block_size': [100], 'blockadam_func': ['ns'],
            # # 'optimizer': ['blockadam'], 'lr': ['5e-5', '3e-5', '1e-5'], 'blockadam_block_size': [32], 'blockadam_func': ['abs'],
            # # 'optimizer': ['mfac'], 'lr': ['1e-3', '1e-4', '1e-5'], 'blockadam_block_size': [32], 'blockadam_func': ['abs'],
        },
    )

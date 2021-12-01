# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import argparse
import random
import math
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.utils import data
import torch.utils.data as Data
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from causal_vae.utils import get_batch_unin_dataset_withlabel, _h_A
import causal_vae.utils as ut
from causal_vae.models.mask_vae import CausalVAE

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x

def save_model_by_name(model_dir, model, global_step):
    save_dir = os.path.join(model_dir, 'checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [S?nderby 2016].
    """

    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

def train_causal_vae(dataset_dir, results_dir, model_dir, epoch_max=101, iter_save=5, run=0, train=True, color=False, toy=None):
    '''
    Train a CausalVAE model
    :param dataset_dir: Directory of saved data
    :param results_dir: Root directory of saved results
    :param model_dir: Root directory of saved model
    :param epoch_max: Number of training epochs
    :param iter_save: Save model every n epochs
    :param run: Run ID. In case you want to run replicates
    :param train: Flag for training
    :param color: Flag for color
    :param toy: Flag for toy
    '''

    layout = [
        ('model={:s}', 'causalvae'),
        ('run={:04d}', run),
        ('color=True', color),
        ('toy={:s}', str(toy))
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    print('Model name:', model_name)
    lvae = CausalVAE(name=model_name, z_dim=16).to(device)
    figs_vae_dir = os.path.join(results_dir, 'figs_vae')
    if not os.path.exists(figs_vae_dir):
        os.makedirs(figs_vae_dir)

    train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 64)
    test_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 1)
    optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
    beta = DeterministicWarmup(n=100, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch

    for epoch in range(epoch_max):
        lvae.train()
        total_loss = 0
        total_rec = 0
        total_kl = 0
        for u, l in train_dataset:
            optimizer.zero_grad()
            # u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
            u = u.to(device)
            L, kl, rec, reconstructed_image, _ = lvae.negative_elbo_bound(u, l, sample=False)

            dag_param = lvae.dag.A

            # dag_reg = dag_regularization(dag_param)
            h_a = _h_A(dag_param, dag_param.size()[0])
            L = L + 3 * h_a + 0.5 * h_a * h_a  # - torch.norm(dag_param)

            L.backward()
            optimizer.step()
            # optimizer.zero_grad()

            total_loss += L.item()
            total_kl += kl.item()
            total_rec += rec.item()

            m = len(train_dataset)
            save_image(u[0], os.path.join(figs_vae_dir, 'reconstructed_image_true_{}.png'.format(epoch)), normalize=True)
            save_image(reconstructed_image[0], os.path.join(figs_vae_dir, 'reconstructed_image_{}.png'.format(epoch)), normalize=True)

        if epoch % 1 == 0:
            print(str(epoch) + ' loss:' + str(total_loss / m) + ' kl:' + str(total_kl / m) + ' rec:' + str(
                total_rec / m) + 'm:' + str(m))

        if epoch % iter_save == 0:
            ut.save_model_by_name(model_dir, lvae, epoch)


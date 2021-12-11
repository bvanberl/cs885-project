#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import time
import random
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils import data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from causal_vae.utils import  get_batch_unin_dataset_withlabel
from causal_vae import utils as ut
import causal_vae.models.mask_vae as sup_dag

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

def inference_causal_vae(dataset_dir, results_dir, model_dir, epoch_max=101, iter_save=5, run=0, train=True, color=False, toy=None, dag='sup_dag'):

    layout = [
        ('model={:s}',  'causalvae'),
        ('run={:04d}', run),
        ('color=True', color),
        ('toy={:s}', str(toy))
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    if dag == "sup_dag":
        lvae = sup_dag.CausalVAE(name=model_name, w=84, h=84, z_dim=36, z1_dim=6, z2_dim=6, inference=True).to(device)
        ut.load_model_by_name(model_dir, lvae, epoch_max - 1)

    figs_vae_dir = os.path.join(results_dir, 'figs_test_vae')
    if not os.path.exists(figs_vae_dir):
        os.makedirs(figs_vae_dir)
    means = torch.zeros(2,3,4).to(device)
    z_mask = torch.zeros(2,3,4).to(device)

    dataset = get_batch_unin_dataset_withlabel(os.path.join(dataset_dir, 'train'), 100)

    count = 0
    sample = False
    print('DAG:{}'.format(lvae.dag.A))
    for u,l in dataset:
        for i in range(6):
            for j in range(-5,5):
                L, kl, rec, reconstructed_image, z_given_dag = lvae.negative_elbo_bound(u.to(device), l.to(device))
                L, kl, rec, reconstructed_image_int,z_given_dag= lvae.negative_elbo_bound(u.to(device),l.to(device),i,sample = sample, adj=-1)  # Before, adj=j*0
            save_image(reconstructed_image_int[0], '{}/reconstructed_image_int_{}_{}.png'.format(figs_vae_dir, i, count),  range = (0,1))
            save_image(reconstructed_image[0], '{}/reconstructed_image_{}_{}.png'.format(figs_vae_dir, i, count),  range=(0, 1))
        save_image(u[0], '{}/figs_test_vae_pendulum/true_{}.png'.format(figs_vae_dir, count))
        count += 1
        if count == 10:
            break

if __name__=='__main__':
    dataset_dir = 'B:/Datasets/School/CausalVAE/cartpole3'
    results_dir = 'results/log/causalvae/cartpole3'
    model_dir = 'results/models/causalvae/cartpole3'
    inference_causal_vae(dataset_dir, results_dir, model_dir)
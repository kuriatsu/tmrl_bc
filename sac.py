import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

from tmrl.util import partial # custom arg tools based on functools.partial
from tmrl.networking import Trainer, RolloutWorker, Server # main base entities

from tmrl.training_offline import TrainingOffline # base training class

import numpy as np
import os

epochs = 50
rounds = 100
steps = 2
start_training = 10
max_training_steps_per_env_step = 10
update_model_interval = 100
update_buffer_interval = 50
device_trainer = "cuda"
memory_size = 10000
batch_size = 100
max_samples_per_episode = 100


memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = "/home/kuriatsu/Source/tmrl_bc/data/"
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

## rtgym environment class
env_vls = cfg_objj.ENV_CLS

device_worker = "cpu"

window_width = 256 # 256-958
window_height = 128 # 128-488

img_width = 64
img_height = 64

img_grayscale = True
imgs_buf_len = cfg.IMG_HIST_LEN # screenshots in each observation
act_buf_len = cfg.ACT_BUF_LEN # number of actions in the action buffer (part of observation)

## MEMORY CLASS
memory_cls = partial(memory_base_cls,
                     memory_size = memory_size,
                     batch_size = batch_size,
                     sample_preprocessor = sample_preprocessor,
                     dataset_path = dataset_path,
                     imgs_obs = imgs_buf_len,
                     act_buf_len = act_buf_len,
                     crc_debug = False)

## CUSTOM MODEL
LOG_STD_MAX = 2
LOG_STD_MIN = -20

from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)

def num_flat_features(x):
    """ computes dimensionality of CNN feature maps when flattened together"
    """
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def conv2d_out_dims(conv_layer, h_in, w_in):
    """ computes the dimensionality of the output in 2D CNN layer
    """
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((h_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out

class VanillaCNN(nn.module):


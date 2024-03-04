import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj

from tmrl.util import partial # custom arg tools based on functools.partial
from tmrl.networking import Trainer, RolloutWorker, Server # main base entities

from tmrl.training_offline import TrainingOffline # base training class

import numpy as np
import os

epochs = 10000 
rounds = 100
steps = 200 
start_training = 1000
max_training_steps_per_env_step = 4.0 
update_model_interval = 200
update_buffer_interval = 200
device_trainer = "cuda"
memory_size = 1000000
batch_size = 256 
max_samples_per_episode = 1000


memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = "/home/kuriatsu/Source/tmrl_bc/data/"
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

## rtgym environment class
env_cls = cfg_obj.ENV_CLS
print(env_cls)

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

class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()
        self.q_net = q_net

        # default input is grayscale 64x64 pix
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        self.flat_features = self.out_channels * self.h_out * self.w_out

        ## MLP input is:
        ## flattened CNN features
        ## speed, gear, RPM (3 floats)
        ## 2 previous actions (2x3 floats)
        ## when critic, selected acttion (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        ## MLP layer
        ## when use the model as policy, we sample from multivariate gaussian defined later, 
        ## thus, the output is 1 dim for critic
        ## output layer for policies is defined later
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        if self.q_net:
            ## the critic takes the next action
            ## act1, 2 are the action in the action buffer (past actions), act is the selected action to update policy
            speed, gear, rpm, images, act1, act2, act = x
        else:
            ## for the policy, the next action is what we are computing
            speed, gear, rpm, images, act1, act2 = x

        ## forwarrd pass of CNN
        ## competition env outputs 4 past images
        ## img is 64x64
        ## stack 4 images along the channel dimention
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(images))
        x = F.relu(self.conv3(images))
        x = F.relu(self.conv4(images))
        
        ## flatten feature map
        flat_features = num_flat_features(x)

        ## output flatten feature map
        x = x.view(-1, flat_features)

        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)

        x = self.mlp(x)
        return x

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct

class MyActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0]
        act_limit = action_space.high[0] # max amplitude of actions

        self.net = VanillaCNN(q_net=False)
        ## the policy output layer, sample actions stochastically in a gaussian
        ## so, average
        self.mu_layer = nn.Linear(256, dim_act)
        ## and log standard deviation
        self.log_std_layer = nn.Linear(256, dim_act)

        ## squash this within the action space with tanh final activation
        self.act_limit = act_limit

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)


    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        print(path)
        with open(path, 'r', encoding="UTF-8", errors="ignore") as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        test : True for test episode and False for training episodes; in SAC, sample randomly during training and deterministidcally at test-time.
        compute logprob: SAC set this to True to retrieve log probabilities
        """
        ## feed obs to MLP
        net_out = self.net(obs)
        ## means of multivariate gauusian:
        mu = self.mu_layer(net_out)
        ## corresponding standard deviation
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        ## sample action in the multivariate gaussian distribution
        pi_distribution = Normal(mu, std)
        if test:
            ## in testing, action is deterministic
            pi_action = mu
        else:
            ## during training, action is samples in the gaussian
            pi_action = pi_distribution.rsample()

        ## retrieve log prob of our gaussian for SAC
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            ## correction formula for TanH squashing
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        ## squash action within action space
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        ## remove batch dimension
        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs,  test=False):
        with torch.no_grad():
            ## no need of log prob
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()


## the critic module for SAC is now simple
class VanillaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        ## since q_net=True, add act to obs
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)

class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.actor = MyActorModule(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)


## CUSTOM TRAINING ALGORITHM
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam

class SACTrainingAgent(TrainingAgent):
    ## no-grad copy of the model used to send the actor weights in get_actor()
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=VanillaCNNActorCritic,
                 gamma=0.99, # discount factor
                 polyak=0.995, # exponential averaging factor for target critic
                 alpha=0.2, # value of the entropy coefficient
                 lr_actor=1e-3, # learning rate for the actor
                 lr_critic=1e-3 # learning rate for the critic
                 ):

        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space,  action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alphah)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        """ 
        execute training iteration from bathced training samples
        Training sample is (o, a, r, o2, d, t)
        o : initial observation
        o2 : final observation of the transition
        r : reward
        d : terminal signal whether o2 is a terminal state
        t : truncated signal whether episode has been truncated by a time limit

        batch :  (o, a, r, o2, d, t)
        """

        o, a, r, o2, d, _ = batch

        ## sample action in the current policy
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        ## compute action-value estimates for the current transition
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        ## compute value target
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        ## critic loss, target and estimate
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        ## optimization step to train critics
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        ## for policy optimization, detach critics from gradient computation graph
        for p in self.q_params:
            p.requires_grad = False

        ## user critics to estimate the value of the action sampled in the current policy
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        ## optimization step to train policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        ## attach critics back into the gradient computation graph
        for p in self.q_params:
            p.requires_grad = True
        
        ## update target model with slowly moving exponential average:
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1-self.polyak)*p.data)
        
        ret_dict = dict(
                loss_actor = loss_pi.detach().item(),
                loss_critic = loss_q.detach().itemm(),
                )

        return dict


training_agent_cls = partial(SACTrainingAgent,
                             model_cls=VanillaCNNActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=0.000005,
                             lr_critic=0.0003)

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)

if __name__ == "__main__":
    import sys

    if sys.argv[1] == "trainer":
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip="55555",
                             server_port="0.0.0.0",
                             password="",
                             security="TLS")
        my_trainer.run()

    elif sys.argv[1] == "worker" or sys.argv[1] == "test":
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=MyActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip="0.0.0.0",
                           server_port="55555",
                           password="",
                           security="TLS",
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=False)
        rw.run()
    elif sys.argv[1] == "server":
        import time
        server = Server(port="55555",
                        password="",
                        security="TLS")

        while True:
            time.sleep(1.0)


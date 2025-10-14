# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec


OBS_KEY = "policy" # ("agents", "observation")
OBS_PRIV_KEY = "priv"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action" # ("agents", "action")
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
TERM_KEY = ("next", "terminated")
DONE_KEY = ("next", "done")
CMD_KEY = "command"


class ResFCLayer(nn.Module):
    def __init__(self, out_dim: int, activation=nn.Mish):
        super().__init__()
        self.linear = nn.LazyLinear(out_dim * 2)
        self.mish = activation()
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x, skip = self.linear(x).chunk(2, dim=-1)
        return self.ln(self.mish(x) + skip)


def make_mlp(num_units, activation=nn.Mish, norm="before", dropout=0.):
    assert norm in ("before", "after", None)
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        elif norm == "after":
            layers.append(activation())
            layers.append(nn.LayerNorm(n))
        else:
            layers.append(activation())
        if dropout > 0. :
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_conv(num_channels, activation=nn.LeakyReLU, kernel_sizes=3, flatten: bool=True):
    layers = []
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(num_channels)
    for n, k in zip(num_channels, kernel_sizes):
        layers.append(nn.LazyConv2d(n, kernel_size=k, stride=2, padding=k//2))
        layers.append(activation())
    if flatten:
        layers.append(nn.Flatten())
    return FlattenBatch(nn.Sequential(*layers), data_dim=3)


class FlattenBatch(nn.Module):
    def __init__(self, module, data_dim: int=1):
        super().__init__()
        self.module = module
        self.data_dim = data_dim

    def forward(self, input: torch.Tensor):
        batch_shape = input.shape[:-self.data_dim]
        flattened_input = input.flatten(0, len(batch_shape)-1)
        output = self.module(flattened_input)
        return output.unflatten(0, batch_shape)


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


class Chunk(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return x.chunk(self.n, dim=-1)

class Duplicate(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return tuple(x for _ in range(self.n))

class Split(nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size
    
    def forward(self, x: torch.Tensor):
        return x.split(self.split_size, dim=-1)


class Actor(nn.Module):
    def __init__(self, action_dim: int, init_noise_scale: float=1.0, predict_std: bool=False, load_noise_scale: float | None=None) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale
    
    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)


class ActorCov(nn.Module):
    """
    Predicts state-dependent covariance between a_t and a_{t-1}.
    """
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean_cov = nn.LazyLinear(action_dim * 2)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.scale_mapping = torch.exp
    
    def forward(self, features: torch.Tensor, prev_action: torch.Tensor, prev_loc: torch.Tensor):
        loc, cov = self.actor_mean_cov(features).chunk(2, dim=-1)
        scale = torch.ones_like(loc) * self.scale_mapping(self.actor_std)
        var = scale.square()
        cov = torch.tanh(cov) * var.detach()
        loc = loc + (cov / var.detach()) * (prev_action - prev_loc)
        var = var - cov.square() / var.detach()
        scale = var.sqrt()
        return loc, scale


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor,
        done: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor,
        discount: torch.Tensor=None
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        nonterm = 1 - terminated.float() # whether to backup value
        nondone = 1 - done.float()       # whether to backup reward
        if discount is None:
            discount = torch.ones_like(nonterm)
        gae = 0
        for step in reversed(range(num_steps)):
            next_value_t = next_value[:, step] * nonterm[:, step]
            gamma_t = discount[:, step] * self.gamma
            delta = reward[:, step] + gamma_t * next_value_t - value[:, step]
            advantages[:, step] = gae = delta + (gamma_t * self.lmbda * nondone[:, step] * gae)
        returns = advantages + value
        return advantages, returns


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 0.01)
        nn.init.constant_(module.bias, 0.)


def compute_policy_loss(
    tensordict: TensorDictBase,
    actor: ProbabilisticActor,
    clip_param: float,
    entropy_coef: float,
    discard_init: bool=True,
):
    dist = actor.get_dist(tensordict)
    log_probs = dist.log_prob(tensordict[ACTION_KEY])
    entropy = dist.entropy()

    adv = tensordict["adv"]
    ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
    surr1 = adv * ratio
    surr2 = adv * ratio.clamp(1. - clip_param, 1. + clip_param)
    policy_loss = torch.min(surr1, surr2)
    if discard_init:
        policy_loss = policy_loss * (~tensordict["is_init"])
    policy_loss = - torch.mean(policy_loss) * dist.event_shape[-1]
    entropy_loss = - entropy_coef * torch.mean(entropy)
    return policy_loss, entropy_loss, entropy.mean()


def compute_value_loss(
    tensordict: TensorDictBase, 
    critic: ModBase,
    clip_param: float,
    critic_loss_fn: nn.Module,
    discard_init: bool=True,
):
    # b_values = tensordict["state_value"]
    b_returns = tensordict["ret"]
    values = critic(tensordict)["state_value"]
    # values_clipped = b_values + (values - b_values).clamp(-clip_param, clip_param)
    # value_loss_clipped = critic_loss_fn(b_returns, values_clipped)
    value_loss_original = critic_loss_fn(b_returns, values)
    # value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

    # mask out first transitions which are generally invalid
    # due to the limiatations of Isaac Sim
    if discard_init:
        value_loss_original = value_loss_original * (~tensordict["is_init"])
    value_loss = value_loss_original.mean()
    explained_var = 1 - value_loss_original.detach() / b_returns.var()

    return value_loss, explained_var


def hard_copy_(source_module: nn.Module, target_module: nn.Module):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.copy_(params_source.data)

def soft_copy_(source_module: nn.Module, target_module: nn.Module, tau: float = 0.01):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.lerp_(params_source.data, tau)


class L2Norm(nn.Module):
    
    def forward(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(1e-7)

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim: int, method="l2"):
        super().__init__()
        self.dim = dim
        if method == "softmax":
            self.f = F.softmax
        elif method == "l2":
            self.f = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(1e-6)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = self.f(x)
        return x.view(*shp)


class ConsistentDropout(nn.Module):
    def __init__(self, p: float, return_mask: bool=True):
        super().__init__()
        self.p = p
        self.scale_factor = 1 / (1- p)
        self.return_mask = return_mask
    
    def forward(self, input: torch.Tensor, mask=None):
        if mask is None:
            mask = input.data.bernoulli(self.p)
        if self.return_mask:
            return input * mask * self.scale_factor, mask
        else:
            return input * mask * self.scale_factor


class MaskWithEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.embedding)
    
    def forward(self, input, mask):
        output = torch.where(mask, self.embedding.expand_as(input), input.detach())
        return output


class NormalExtractor(nn.Module):
    def __init__(self, include_loc: bool=True, num_samples: int = 1):
        super().__init__()
        self.include_loc = include_loc
        self.num_sample = num_samples

    def forward(self, x: torch.Tensor):
        x_loc, x_scale = x.chunk(2, -1)
        x_sample = x_loc.unsqueeze(-2).expand(*x_loc.shape[:-1], self.num_sample, -1)
        x_sample = x_sample + torch.randn_like(x_sample) * x_scale.exp().unsqueeze(-2)
        if self.include_loc:
            x_sample = torch.cat([x_sample, x_loc.unsqueeze(-2)], dim=-2)
        return x_sample.flatten(-2), x_loc, x_scale


class CatTensors(ModBase):
    def __init__(self, in_keys, out_key, del_keys=False, sort=True):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [out_key]

        self.del_keys = del_keys
        self.sort = sort
        if self.sort:
            self.in_keys = sorted(self.in_keys)

    def forward(self, tensordict: TensorDictBase):
        out = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)
        tensordict.set(self.out_keys[0], out)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict


def collect_info(infos, prefix=""):
    return {prefix+k: v.mean().item() for k, v in torch.stack(infos).items()}


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)


def parse_keys(spec: CompositeSpec, keys: list[str]):
    """
    Parse the keys into `mlp_keys`, `cnn_keys`, and `aux_keys`.
    Keys ending with "_" are considered auxiliary keys.

    """
    mlp_keys = []
    cnn_keys = []
    aux_keys = []
    
    for key in keys:
        if key in spec.keys(True, True):
            _spec = spec[key]
        if key.endswith("_"):
            aux_keys.append(key)
            continue
        if _spec.ndim == 2:
            mlp_keys.append(key)
        else:
            cnn_keys.append(key)
    return mlp_keys, cnn_keys, aux_keys

"""Actor-critic around the BC policy (the brief's "Shared PPO fine-tuning setup").

- Actor: a trainable copy of the whole BC network. Its six outputs (absolute cursor position in
  MOVE_SCALE units, two press logits, two held logits) are the mean of a Gaussian with a learned,
  state-independent log-std. The cursor latents are what the brief asks for; the key latents get
  Gaussian logit noise too, decoded by the BC click rule (osu_sim/key_decoder.py), instead of
  Bernoulli-sampled clicks: a sampled click every frame the press probability is above zero would
  break the edge-triggered decoder (deviation, the experiment log). Deterministic play = the mean = BC exactly.
- Critic: its own encoder, a copy of the BC encoder (so value gradients never touch the actor's
  features), and a small value head.
- Anchor: a frozen copy of BC; KL(actor || BC with the initial std), weighted by a coefficient the
  trainer decays.
"""
import copy
import math

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from models.heads import MOVE_SCALE


def means(model, frames: torch.Tensor) -> torch.Tensor:
    out = model(frames)
    return torch.cat([out["position"], out["press"], out["held"]], dim=1).float()


class Agent(nn.Module):
    def __init__(self, bc: nn.Module, aim_std_px: float, key_std: float, features: int = 512):
        super().__init__()
        if bc.aim != "absolute" or bc.prev_action:
            raise ValueError("PPO expects the absolute-aim BC baseline without previous actions")
        self.actor = copy.deepcopy(bc)
        self.reference = copy.deepcopy(bc).requires_grad_(False).eval()
        self.critic = copy.deepcopy(bc.backbone)
        self.value_head = nn.Sequential(nn.Linear(features, 256), nn.ReLU(), nn.Linear(256, 1))
        std = torch.tensor([aim_std_px / MOVE_SCALE] * 2 + [key_std] * 4)
        self.log_std = nn.Parameter(std.log())
        self.register_buffer("reference_std", std.clone())

    def std(self) -> torch.Tensor:
        return self.log_std.clamp(math.log(1e-3), math.log(2.0)).exp()

    def dist(self, frames: torch.Tensor) -> Normal:
        mu = means(self.actor, frames)
        return Normal(mu, self.std().expand_as(mu))

    def value(self, frames: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic(frames)).squeeze(-1).float()

    def kl_to_reference(self, frames: torch.Tensor, dist: Normal) -> torch.Tensor:
        with torch.no_grad():
            ref = Normal(means(self.reference, frames), self.reference_std.expand_as(dist.mean))
        return kl_divergence(dist, ref).sum(-1)

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.log_std]

    def critic_parameters(self):
        return list(self.critic.parameters()) + list(self.value_head.parameters())

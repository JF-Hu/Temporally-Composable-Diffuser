import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import d4rl


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms, v_min, v_max):
        super(Critic, self).__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, num_atoms)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x = F.relu(self.layer_1(xu))
        x = F.relu(self.layer_2(x))
        x = F.softmax(self.layer_3(x), dim=-1)
        return x


class Categorical_Q_Function(nn.Module):
    def __init__(self,
                 state_dim, action_dim, max_action,
                 num_atoms, v_min, v_max,
                 discount=0.99, ema_decay=0.995, actor_lr=1e-4, critic_lr=1e-4,
                 target_network_update_freq=10,
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.discount = discount
        self.ema_decay = ema_decay
        self.target_network_update_freq = target_network_update_freq
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.target_z = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim, num_atoms, v_min, v_max)

        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_critic = Critic(state_dim, action_dim, num_atoms, v_min, v_max)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    # Soft update target network
    def update_target_network_parameters(self, policy_net, target_net):
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_((1 - self.ema_decay) * param.data + self.ema_decay * target_param.data)

    def train_model(self, batch, train_times):
        state_batch = batch.trajectories[:, :, self.action_dim:]
        state_batch = torch.reshape(state_batch, [-1, state_batch.shape[-1]])
        action_batch = batch.trajectories[:, :, :self.action_dim]
        action_batch = torch.reshape(action_batch, [-1, action_batch.shape[-1]])
        next_state_batch = batch.next_obs
        next_state_batch = torch.reshape(next_state_batch, [-1, next_state_batch.shape[-1]])
        # action_batch = batch.trajectories[:, 1:, :self.action_dim]
        reward_batch = batch.rewards
        reward_batch = torch.reshape(reward_batch, [-1, reward_batch.shape[-1]])
        done_batch = batch.dones
        done_batch = torch.reshape(done_batch, [-1, done_batch.shape[-1]])
        # todo torch.reshape(state_batch[:, 1:, :], [-1,state_batch.shape[-1]])

        with torch.no_grad():
            next_action_batch = self.target_actor(next_state_batch)
            target_probs = self.target_critic(next_state_batch, next_action_batch)
            target_z = self.target_z.to(state_batch.device)
            target_z_ = reward_batch + (1 - done_batch) * self.discount * target_z.view(1, -1)
            target_z_.clamp_(self.v_min, self.v_max)
            b = (target_z_ - self.v_min) / self.delta_z
            l = b.floor().clamp(0, self.num_atoms - 1)
            u = b.ceil().clamp(0, self.num_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * target_probs
            d_m_u = (b - l) * target_probs
            target_probs_ = torch.zeros_like(target_probs)
            # for i in range(target_probs_.size(0)):
            #     target_probs_[i].index_add_(0, l[i].long(), d_m_l[i])
            #     target_probs_[i].index_add_(0, u[i].long(), d_m_u[i])
            target_probs_.scatter_add_(dim=1, index=l.int().to(torch.long), src=d_m_l)
            target_probs_.scatter_add_(dim=1, index=u.int().to(torch.long), src=d_m_u)

        # Update the critic
        self.critic_optimizer.zero_grad()
        current_probs = self.critic(state_batch, action_batch)
        loss_critic = (-(target_probs_ * current_probs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Update the actor
        self.actor_optimizer.zero_grad()
        loss_actor = (-(self.critic(state_batch, self.actor(state_batch)) * torch.linspace(self.v_min, self.v_max, self.num_atoms).to(state_batch.device)).sum(-1)).mean()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Update target networks
        if train_times % self.target_network_update_freq == 0:
            self.update_target_network_parameters(self.actor, self.target_actor)
            self.update_target_network_parameters(self.critic, self.target_critic)

        return {"dist_critic_loss": loss_critic, "dist_actor_loss": loss_actor}










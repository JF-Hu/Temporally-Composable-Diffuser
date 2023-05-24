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

env = gym.make('hopper-medium-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

num_atoms = 101
v_min = -5
v_max = 100
delta_z = (v_max - v_min) / (num_atoms - 1)

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


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)

# Soft update target network
def update_target_network_parameters(policy_net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train(replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.005):
    for _ in range(iterations):
        # Sample a batch of experiences from the replay buffer
        samples = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor([sample[0] for sample in samples]).to(device)
        action_batch = torch.FloatTensor([sample[1] for sample in samples]).to(device)
        next_state_batch = torch.FloatTensor([sample[2] for sample in samples]).to(device)
        reward_batch = torch.FloatTensor([sample[3] for sample in samples]).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor([sample[4] for sample in samples]).unsqueeze(1).to(device)

        # Compute the target distribution
        with torch.no_grad():
            next_action_batch = target_actor(next_state_batch)
            target_probs = target_critic(next_state_batch, next_action_batch)
            target_z = torch.linspace(v_min, v_max, num_atoms).to(device)
            target_z_ = reward_batch + (1 - done_batch) * discount * target_z.view(1, -1)
            target_z_.clamp_(v_min, v_max)
            b = (target_z_ - v_min) / delta_z
            l = b.floor().clamp(0, num_atoms - 1)
            u = b.ceil().clamp(0, num_atoms - 1)
            # target_probs_ = (u + (l == u).float() - b) * target_probs
            # target_probs_ = target_probs_.gather(1, l.unsqueeze(2).repeat(1, 1, num_atoms))
            # target_probs_ += (b - l) * target_probs
            # target_probs_ = target_probs_.gather(1, u.unsqueeze(2).repeat(1, 1, num_atoms))

            d_m_l = (u + (l == u).float() - b) * target_probs
            d_m_u = (b - l) * target_probs
            target_probs_ = torch.zeros_like(target_probs)
            for i in range(target_probs_.size(0)):
                target_probs_[i].index_add_(0, l[i].long(), d_m_l[i])
                target_probs_[i].index_add_(0, u[i].long(), d_m_u[i])

        # Update the critic
        critic_optimizer.zero_grad()
        current_probs = critic(state_batch, action_batch)
        # loss_critic = -1 * (target_probs_ * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()
        loss_critic = (-(target_probs_ * current_probs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        loss_critic.backward()
        critic_optimizer.step()

        # Update the actor
        actor_optimizer.zero_grad()
        # loss_actor = -1 * (critic(state_batch, actor(state_batch)) * torch.log(current_probs + 1e-8)).sum(dim=2).mean()
        loss_actor = (-(critic(state_batch, actor(state_batch)) * torch.linspace(v_min, v_max, num_atoms).to(device)).sum(-1)).mean()
        loss_actor.backward()
        actor_optimizer.step()

        # Update target networks
        update_target_network_parameters(actor, target_actor, tau)
        update_target_network_parameters(critic, target_critic, tau)

actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim, num_atoms, v_min, v_max).to(device)

target_actor = Actor(state_dim, action_dim, max_action).to(device)
target_critic = Critic(state_dim, action_dim, num_atoms, v_min, v_max).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

update_target_network_parameters(actor, target_actor, 1.0)
update_target_network_parameters(critic, target_critic, 1.0)


max_episodes = 500
max_timesteps = 1000
exploration_noise = 0.1
reward_scale = 40

replay_buffer = ReplayBuffer()
exploration_decay = 1.0

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()

        # Add exploration noise
        if episode > 200:
            exploration_decay = (episode - 200)
        action = (action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])).clip(
            env.action_space.low, env.action_space.high
        )

        next_state, reward, done, _ = env.step(action)
        reward = reward / reward_scale

        # Store experience in the replay buffer
        replay_buffer.add((state, action, next_state, reward, float(done)))

        state = next_state
        episode_reward += reward

        if done:
            break

    # Train the networks
    if len(replay_buffer.storage) >= 1000:
        train(replay_buffer, 100)

    print(f"Episode {episode + 1}: eval_time_step:{t}, {episode_reward * reward_scale}")


from collections import namedtuple
import numpy as np
import torch
import pdb
import copy
import math
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, get_sequence_dataset_size
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
GoalRewardBatch = namedtuple('Batch', 'trajectories conditions returns goals observations next_observations resampled_goals bisimulation_rewards')
ReturnToGoBatch = namedtuple('Batch', 'trajectories conditions returns_timesteps')
ActionRewardReturnToGoBatch = namedtuple('Batch', 'trajectories conditions actionrewards_returns_timesteps')
FFTReturnToGoBatch = namedtuple('Batch', 'fft_trajectories fft_conditions returns_timesteps trajectories conditions')
FirstPosReturnToGoBatch = namedtuple('Batch', 'trajectories conditions returns_timesteps_firstpos')
DistQvalueReturnToGoBatch = namedtuple('Batch', 'trajectories conditions returns dones rewards next_obs')
JointDistFARAndHC = namedtuple('Batch', 'trajectories conditions returns_timesteps FARSequence')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False,
        include_goal_returns=False, goal_distance=1, return_type=None, traj_length_must_bigger_than=None, top_K_length=5,
        traj_return_must_bigger_than=None, history_length=1, distance_to_failure_obs=0):
        self.dataset_name = env
        self.env_name = env
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        if self.dataset_name.split("-")[0] not in ["halfcheetah", "hopper", "walker2d"]:
            max_n_episodes, max_path_length = get_sequence_dataset_size(env)
        elif self.dataset_name.split("-")[1] == "random":
            max_n_episodes, max_path_length = get_sequence_dataset_size(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        if return_type == 8 or return_type == 7:
            max_path_length = max_path_length + history_length - 1
        else:
            assert history_length == 1
        self.max_path_length = max_path_length
        if return_type == 1:
            self.discount = 0.999
        else:
            self.discount = discount
        if self.dataset_name.split("-")[1] in ["prospective_condition", "immediate_condition"]:
            self.discount = 1.0
        if return_type == 7:
            termination_penalty = 0

        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.unify_discounts = 1.00 ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        self.termination_penalty = termination_penalty
        self.include_goal_returns = include_goal_returns
        self.goal_distance = goal_distance
        self.return_type = return_type
        self.top_K_length = top_K_length
        self.history_length = history_length
        self.distance_to_failure_obs = distance_to_failure_obs
        try:
            self.hard_reward_return_scale = (1 - self.discount ** (self.max_path_length - 1)) / (1 - self.discount)
        except:
            self.hard_reward_return_scale = self.max_path_length - 1

        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)

        for i, episode in enumerate(itr):
            if return_type == 6:
                fields.add_path(episode, discounts=self.discounts)
            elif return_type == 7:
                fields.add_path(episode, history_length=history_length, discounts=self.unify_discounts)
            elif return_type == 8:
                fields.add_path(episode, history_length=history_length)
            else:
                fields.add_path(episode)
        fields.finalize()
        self.fields = fields

        # if "halfcheetah" in self.env_name.split("-"):
        #     self.reshape_reward()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon, goal_distance)
        if self.include_goal_returns:
            self.bound_up, self.bound_down = self.get_goal_return_bound(goal_distance=goal_distance)
        self.min_trajectory_return, self.max_trajectory_return, self.all_traj_return_up_bound, self.all_traj_return_down_bound = self.get_max_min_trajectory_return()
        self.max_step_rewards, self.min_step_rewards, self.best_traj_step_rewards, self.min_single_step_reward, self.max_single_step_reward = self.get_every_timestep_reward_situation()
        self.normed_zero_reward_value = (0 - self.min_single_step_reward) / (self.max_single_step_reward - self.min_single_step_reward)
        if return_type == 6 or return_type == 7:
            self.max_discounted_trajectory_return, self.min_discounted_trajectory_return = np.max(self.fields._dict['discounted_returns']), np.min(self.fields._dict['discounted_returns'])
            self.reward_return_scale = self.max_discounted_trajectory_return - self.min_discounted_trajectory_return
        else:
            self.reward_return_scale = 0

        if self.return_type == 0:
            self.normed_max_discounted_return, self.normed_min_discounted_return = self.calculate_max_and_min_env_discounted_return()

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def get_max_min_discounted_return(self):
        return np.max(self.fields._dict['discounted_returns']), np.min(self.fields._dict['discounted_returns'])

    def get_positional_encoding(self, sequence_len, data_dimention):
        position = np.expand_dims(np.arange(0, sequence_len, dtype=np.float32), axis=1)
        div_term = np.exp(np.arange(0, data_dimention, 2) * (-math.log(10000.0) / data_dimention))
        pos_enc = np.zeros((sequence_len, data_dimention))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        if data_dimention % 2 == 1:
            pos_enc[:, 1::2] = np.cos(position * div_term[0:-1])
        else:
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def reshape_reward(self):
        min_rewrd = np.min(self.fields.rewards)
        if min_rewrd >= 0:
            return
        min_rewrd = np.abs(min_rewrd)
        for ep_i in range(len(self.fields.path_lengths)):
            self.fields.rewards[ep_i] += np.expand_dims(np.concatenate([min_rewrd * np.ones(self.fields['path_lengths'][ep_i]), np.zeros(self.max_path_length - self.fields['path_lengths'][ep_i])], axis=-1), axis=-1)

    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        '''
            'fft_observations'
            normalize fields that will be predicted by the diffusion model
        '''
        if self.dataset_name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
            keys = ['observations', 'actions']
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon, goal_distance):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        if not self.include_goal_returns:
            for i, path_length in enumerate(path_lengths):
                max_start = min(path_length - 1, self.max_path_length - horizon)
                if not self.use_padding:
                    max_start = min(max_start, path_length - horizon)
                if max_start - int(self.distance_to_failure_obs) > 0:
                    max_start -= int(self.distance_to_failure_obs)
                for start in range(max_start):
                    end = start + horizon
                    indices.append((i, start, end))
        else:
            for i, path_length in enumerate(path_lengths):
                min_diff_start = 0
                min_goal_start = 0
                max_diff_start = min(path_length - 1, self.max_path_length - horizon)
                max_goal_start = min(path_length - 1, self.max_path_length - goal_distance)
                if not self.use_padding:
                    max_diff_start = min(max_diff_start, path_length - horizon)
                    max_goal_start = min(max_goal_start, path_length - goal_distance)
                for start in range(np.maximum(min_diff_start, min_goal_start), np.minimum(max_diff_start, max_goal_start), 1):
                    diff_end = start + horizon
                    goal_end = start + goal_distance
                    indices.append((i, start, diff_end, goal_end))
        indices = np.array(indices)
        return indices

    def make_goal_indices(self, path_lengths, goal_distance):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            min_start = 10
            max_start = min(path_length - 1, self.max_path_length - goal_distance)
            if not self.use_padding:
                max_start = min(max_start, path_length - goal_distance)
            for start in range(min_start, max_start, step=1):
                end = start + goal_distance
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_max_min_trajectory_return(self):
        dataset_rewards = np.reshape(self.fields._dict['rewards'], [self.fields._count, -1])
        positive_flags = np.where(dataset_rewards>=0, 1, 0)
        negative_flags = np.where(dataset_rewards<=0, 1, 0)
        max_reward_traj_idx = np.argmax(np.sum(dataset_rewards, axis=-1))
        max_trajectory_return = np.max(np.sum(dataset_rewards * positive_flags, axis=-1))
        min_trajectory_return = np.min(np.sum(dataset_rewards * negative_flags, axis=-1))
        all_traj_return_up_bound = np.sum(dataset_rewards * positive_flags, axis=-1)
        all_traj_return_down_bound = np.sum(dataset_rewards * negative_flags, axis=-1)
        return min_trajectory_return, max_trajectory_return, all_traj_return_up_bound, all_traj_return_down_bound

    def get_every_timestep_reward_situation(self, use_top_K=True):
        dataset_rewards = np.reshape(copy.deepcopy(self.fields._dict['rewards']), [self.fields._count, -1])
        ep_rewrds = np.sum(dataset_rewards, axis=-1)
        ep_reward_traj_idx = []
        for ep_idx, ep_reward in enumerate(ep_rewrds):
            ep_reward_traj_idx.append((ep_idx, ep_reward))
        ep_reward_traj_idx = np.vstack(ep_reward_traj_idx)
        # todo for all trajectories, we sorted the trajectories
        sorted_rewards_wst_return = ep_reward_traj_idx[np.argsort(ep_reward_traj_idx[:, 1])]
        if use_top_K:
            length = self.top_K_length
            ep_idx = (sorted_rewards_wst_return[-length:, 0]).astype(np.int32)
            terminal_flags = []
            for ep_i in range(length):
                terminal_flags.append(np.concatenate([np.ones(self.fields['path_lengths'][ep_idx[ep_i]]), np.zeros(self.max_path_length - self.fields['path_lengths'][ep_idx[ep_i]])], axis=-1))
            terminal_flags = np.vstack(terminal_flags)
            best_traj_step_rewards = np.sum(dataset_rewards[ep_idx], axis=0) / (np.sum(terminal_flags, axis=0) + 1e-6)
        else:
            best_traj_step_rewards = np.mean(dataset_rewards[(sorted_rewards_wst_return[-1:, 0]).astype(np.int32)], axis=0)
        # todo we remove the termination_penalty to get the environmental rewars
        for ep_i in range(len(dataset_rewards)):
            if self.fields['terminals'][ep_i].any():
                dataset_rewards[ep_i, self.fields['path_lengths'][ep_i]-1] -= self.termination_penalty
        max_step_rewards = np.max(dataset_rewards, axis=0)
        for ep_i in range(len(dataset_rewards)):
            if self.fields['terminals'][ep_i].any():
                done_flags = np.concatenate([np.ones(self.fields['path_lengths'][ep_i]), np.zeros(self.max_path_length - self.fields['path_lengths'][ep_i])], axis=-1)
                min_flags = np.concatenate([np.zeros(self.fields['path_lengths'][ep_i]), 999999 * np.ones(self.max_path_length - self.fields['path_lengths'][ep_i])], axis=-1)
                dataset_rewards[ep_i] = dataset_rewards[ep_i] * done_flags
                dataset_rewards[ep_i] = dataset_rewards[ep_i] + min_flags
        min_step_rewards = np.min(dataset_rewards, axis=0)
        for idx in range(len(min_step_rewards)):
            if np.abs(max_step_rewards[idx] - min_step_rewards[idx] < 1e-6):
                min_step_rewards[idx] -= 0.1
                max_step_rewards[idx] += 0.025
        min_single_step_reward = np.min(min_step_rewards)
        max_single_step_reward = np.max(max_step_rewards)
        return max_step_rewards, min_step_rewards, best_traj_step_rewards, min_single_step_reward, max_single_step_reward

    def get_goal_return_bound(self, goal_distance, weather_discount=False):
        bound_up, bound_down = 0., 0.
        for i, path_length in enumerate(self.fields.path_lengths):
            max_start = min(path_length - 1, self.max_path_length - goal_distance)
            if not self.use_padding:
                max_start = min(max_start, path_length - goal_distance)
            for start in range(max_start):
                end = start + goal_distance
                goal_rewards = self.fields.rewards[i, start:end]
                if weather_discount:
                    discounts = self.discounts[:len(goal_rewards)]
                    goal_rewards = (discounts * goal_rewards).sum()
                else:
                    goal_rewards = goal_rewards.sum()
                if bound_up < goal_rewards:
                    bound_up = goal_rewards
                if bound_down > goal_rewards:
                    bound_down = goal_rewards
        return bound_up, bound_down

    def calculate_max_and_min_env_discounted_return(self):
        max_discounted_return, min_discounted_return = 0.0, 0.0
        for indice in self.indices:
            path_ind, start, end = indice
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            max_discounted_return = max_discounted_return if max_discounted_return > returns else returns
            min_discounted_return = min_discounted_return if min_discounted_return < returns else returns
        print(f"[current dataset] max_discounted_return:{max_discounted_return}, normed_max_discounted_return:{max_discounted_return/self.returns_scale}, min_discounted_return:{min_discounted_return}, normed_min_discounted_return:{min_discounted_return/self.returns_scale}")
        return max_discounted_return/self.returns_scale, min_discounted_return/self.returns_scale

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        if self.include_goal_returns:
            path_ind, start, diff_end, goal_end = self.indices[idx]

            observations = self.fields.normed_observations[path_ind, start:diff_end]
            goal_prediction_observations = self.fields.normed_observations[path_ind, start]
            goal_prediction_next_observations = self.normalizer.normalize(self.fields.next_observations[path_ind, start], key='observations')
            actions = self.fields.normed_actions[path_ind, start:diff_end]
            goals = self.fields.normed_observations[path_ind, goal_end]
            goal_prediction_goals = self.normalizer.normalize(self.fields.observations[path_ind, goal_end], key='observations')

            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1)

            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            # bisimulation_rewards = (discounts * rewards).sum()
            bisimulation_rewards = ((self.fields.rewards[path_ind, start:goal_end]).sum() - self.bound_down)/(self.bound_up - self.bound_down)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = GoalRewardBatch(trajectories, conditions, returns, goals, goal_prediction_observations, goal_prediction_next_observations, goal_prediction_goals, bisimulation_rewards)
        else:
            path_ind, start, end = self.indices[idx]

            observations = self.fields.normed_observations[path_ind, start:end]
            actions = self.fields.normed_actions[path_ind, start:end]

            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1)
            if self.return_type == 9:
                fft_observations = self.fields.normed_fft_observations[path_ind, start:end]
                fft_conditions = self.get_conditions(fft_observations)
                fft_trajectories = np.concatenate([actions, fft_observations], axis=-1)
            if self.include_returns:
                rewards = self.fields.rewards[path_ind, start:]
                if self.return_type == 0:    # todo normal return
                    discounts = self.discounts[:len(rewards)]
                    returns = (discounts * rewards).sum()
                    returns = np.array([returns / self.returns_scale], dtype=np.float32)
                    batch = RewardBatch(trajectories, conditions, returns)
                elif self.return_type == 1:  # todo hard reward
                    positive_hard_rewards = np.where(rewards > 0, 1, 0)
                    negative_hard_rewards = np.where(rewards < 0, -1, 0)
                    rewards = positive_hard_rewards + negative_hard_rewards
                    discounts = self.discounts[:len(rewards)]
                    returns = (discounts * rewards).sum()
                    # returns_scale = (1 - self.discount ** (self.max_path_length -1))/(1 - self.discount)
                    returns = np.array([returns/self.hard_reward_return_scale], dtype=np.float32)
                    batch = RewardBatch(trajectories, conditions, returns)
                elif self.return_type == 2 or self.return_type == 3:  # todo RR-TCD AND RQR-TCD
                    returns = (rewards.sum() - self.min_trajectory_return) / (self.max_trajectory_return - self.min_trajectory_return)
                    first_step_reward = (np.squeeze(self.fields.rewards[path_ind, start]) - self.min_single_step_reward) / (self.max_single_step_reward - self.min_single_step_reward)
                    batch = ActionRewardReturnToGoBatch(trajectories, conditions, np.array((first_step_reward, returns, start / self.max_path_length)))
                elif self.return_type == 9:  # todo fft_observation
                    returns = (rewards.sum() - self.min_trajectory_return) / (self.max_trajectory_return - self.min_trajectory_return)
                    first_step_reward = (np.squeeze(self.fields.rewards[path_ind, start]) - self.min_single_step_reward) / (self.max_single_step_reward - self.min_single_step_reward)
                    batch = FFTReturnToGoBatch(fft_trajectories, fft_conditions, np.array((first_step_reward, returns, start / self.max_path_length)), trajectories, conditions)
                elif self.return_type == 4:  # todo  TFD
                    returns = (rewards.sum() - self.min_trajectory_return) / (self.max_trajectory_return - self.min_trajectory_return)
                    batch = ReturnToGoBatch(trajectories, conditions, np.array((returns, start / self.max_path_length)))
                elif self.return_type == 5:  # todo distributional_q RTG hard_reward
                    positive_hard_rewards = np.where(rewards > 0, 1, 0)
                    negative_hard_rewards = np.where(rewards < 0, -1, 0)
                    rewards = positive_hard_rewards + negative_hard_rewards
                    discounts = self.discounts[:len(rewards)]
                    returns = (discounts * rewards).sum()
                    returns = np.array([returns / self.hard_reward_return_scale], dtype=np.float32)
                    dones = self.fields.terminals[path_ind, start:end]
                    next_norm_state = self.fields.normed_next_observations[path_ind, start:end]
                    batch = DistQvalueReturnToGoBatch(trajectories, conditions, returns, dones, rewards[:end-start], next_norm_state)
                elif self.return_type == 6:  # todo distributional_q RTG DQD
                    returns = self.fields.discounted_returns[path_ind, start]
                    returns = (returns / self.reward_return_scale).astype(np.float32)
                    dones = self.fields.terminals[path_ind, start:end]
                    next_norm_state = self.fields.normed_next_observations[path_ind, start:end]
                    batch = DistQvalueReturnToGoBatch(trajectories, conditions, returns, dones, rewards[:end-start], next_norm_state)
                elif self.return_type == 7:  # todo joint distribution of state sequence, reward sequence under the history condition and RTG
                    reward_sequence = (self.fields.rewards[path_ind, start:end] - self.min_single_step_reward) / (self.max_single_step_reward - self.min_single_step_reward)
                    rewards = self.fields.rewards[path_ind, start + self.history_length - 1:]
                    returns = (rewards.sum() - self.min_trajectory_return) / (self.max_trajectory_return - self.min_trajectory_return)
                    batch = JointDistFARAndHC(trajectories, conditions, np.array((returns, start / self.max_path_length)), np.reshape(reward_sequence, [-1, 1]))
                elif self.return_type == 8:  # todo TCD
                    rewards = self.fields.rewards[path_ind, start+self.history_length-1:]
                    returns = (rewards.sum() - self.min_trajectory_return) / (self.max_trajectory_return - self.min_trajectory_return)
                    first_step_reward = (np.squeeze(self.fields.rewards[path_ind, start+self.history_length-1]) - self.min_single_step_reward) / (self.max_single_step_reward - self.min_single_step_reward)
                    batch = ActionRewardReturnToGoBatch(trajectories, conditions, np.array((first_step_reward, returns, start / self.max_path_length)))
                else:
                    raise Exception("Config.return_type is set wrong value.")
            else:
                batch = Batch(trajectories, conditions)

        return batch

class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

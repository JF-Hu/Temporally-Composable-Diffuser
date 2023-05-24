import os
import copy
import numpy as np
import torch
import einops
import pdb
import time
import diffuser
from copy import deepcopy
import wandb
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        general_goal_prediction=False,
        diffusion_goal_prediction=False,
        return_type=None,
        wandb_log=False,
        wandb_log_frequency=100,
        wandb_project_name=None,
        step_reward_prediction_model=None,
        history_length=1,
        save_range=[300000, 400000],
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        if return_type == 9:
            self.state_predict_model = copy.deepcopy(self.model)
            self.ema_state_predict_model = copy.deepcopy(self.model)
            self.state_predict_optimizer = torch.optim.Adam(self.state_predict_model.parameters(), lr=train_lr)


        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.general_goal_prediction = general_goal_prediction
        self.diffusion_goal_prediction = diffusion_goal_prediction
        self.return_type = return_type
        self.wandb_log = wandb_log
        self.wandb_log_frequency = wandb_log_frequency
        self.wandb_project_name = wandb_project_name
        self.step_reward_prediction_model = step_reward_prediction_model
        self.history_length = history_length
        self.save_range = save_range
        assert general_goal_prediction + diffusion_goal_prediction == 1

        self.reset_parameters()
        self.step = 0

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        if self.return_type == 9:
            self.ema_state_predict_model.load_state_dict(self.state_predict_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
        if self.return_type == 9:
            self.ema.update_model_average(self.ema_state_predict_model, self.state_predict_model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train_bisimulation_model(self, batch_data):
        observations = torch.reshape(batch_data.observations, [-1, batch_data.observations.shape[-1]])
        next_observations = torch.reshape(batch_data.next_observations, [-1, batch_data.next_observations.shape[-1]])
        goals = torch.reshape(batch_data.resampled_goals, [-1, batch_data.resampled_goals.shape[-1]])
        rewards = torch.reshape(batch_data.bisimulation_rewards, [-1, 1])
        shulffle_idx = torch.randperm(observations.shape[0])

        states_i = observations
        states_j = observations[shulffle_idx]
        rewards_i = rewards
        rewards_j = rewards[shulffle_idx]
        next_states_i = next_observations
        next_states_j = next_observations[shulffle_idx]
        goals_i = goals
        goals_j = goals[shulffle_idx]
        loss_info = self.model.bisimulation_model.train_model(
            states_i, states_j, rewards_i, rewards_j, next_states_i, next_states_j, goals_i, goals_j)
        return loss_info

    def train_return_based_state_to_goal_model(self, batch_data):
        observations = torch.reshape(batch_data.observations, [-1, batch_data.observations.shape[-1]])
        goals = torch.reshape(batch_data.resampled_goals, [-1, batch_data.resampled_goals.shape[-1]])
        rewards = torch.reshape(batch_data.bisimulation_rewards, [-1, 1])
        loss_info = self.model.goal_prediction_model.train_model(states_i=observations, rewards_i=rewards, goals_i=goals)
        return loss_info

    def train(self, n_train_steps, n_epochs=0):
        if self.wandb_log:
            wandb.init(project=self.wandb_project_name)
        timer = Timer()
        for step in range(n_train_steps):
            if self.return_type == 8:
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                    batch.conditions.clear()
                    batch.conditions.update({self.history_length-1: batch.trajectories[:, 0:self.history_length, self.model.action_dim:]})
                    loss, infos = self.model.loss(*batch)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
            elif self.return_type == 7:
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                    batch = type(batch)(torch.cat([batch.trajectories, batch.FARSequence], dim=-1), batch.conditions, batch.returns_timesteps, batch.FARSequence)
                    batch.conditions.clear()
                    batch.conditions.update({self.history_length-1: batch.trajectories[:, 0:self.history_length, self.model.action_dim:]})
                    # batch.trajectories = torch.cat([batch.trajectories, batch.FARSequence], dim=-1)
                    loss, infos = self.model.loss(x=batch.trajectories, cond=batch.conditions, returns=batch.returns_timesteps)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
            elif self.return_type == 5 or self.return_type == 6:
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                    loss, infos = self.model.loss(x=batch.trajectories, cond=batch.conditions, returns=batch.returns)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    reward_prediction_loss = self.step_reward_prediction_model.train_model(batch=batch, train_times=n_epochs*n_train_steps + step)
                    infos.update(reward_prediction_loss)
            else:
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                    if self.return_type == 9:
                        fft_trajectories, fft_conditions, returns_timesteps, trajectories, conditions = batch.fft_trajectories, batch.fft_conditions, batch.returns_timesteps, batch.trajectories, batch.conditions
                        loss, infos = self.model.loss(x=fft_trajectories, cond=fft_conditions, returns=returns_timesteps, goals=None, state_predict_x=trajectories)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()

                        state_predict_loss, state_predict_infos = self.state_predict_model.loss(x=trajectories, cond=conditions, returns=returns_timesteps, goals=None, state_predict_x=trajectories)
                        state_predict_loss = state_predict_loss / self.gradient_accumulate_every
                        state_predict_loss.backward()
                        for key, val in state_predict_infos.items():
                            infos.update({f"state_predict_{key}": val})
                        infos.update({f"state_predict_loss": state_predict_loss})
                    else:
                        loss, infos = self.model.loss(*batch)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                        if self.model.goals_condition:
                            if self.general_goal_prediction:
                                state_to_goal_loss_info = self.train_return_based_state_to_goal_model(batch_data=batch)
                                # bisimulation_loss_info = self.model.bisimulation_model.train_bisimulation_model(batch_data=batch)
                                infos.update(state_to_goal_loss_info)
                        if self.return_type == 2 or self.return_type == 3:
                            reward_prediction_loss = self.step_reward_prediction_model.train_model(batch)
                            infos.update(reward_prediction_loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.return_type == 9:
                self.state_predict_optimizer.step()
                self.state_predict_optimizer.zero_grad()

            if self.wandb_log:
                if self.step % self.wandb_log_frequency == 0:
                    wandb_infos = {}
                    wandb_infos.update(infos)
                    wandb_infos.update({"diffusion_loss": loss})
                    for info_key, info_val in wandb_infos.items():
                        wandb_infos[info_key] = info_val.detach().cpu().numpy()
                    wandb.log(wandb_infos, step=self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                if self.step >= self.save_range[0] and self.step <= self.save_range[1] and self.step % 100000 == 0:
                    self.save_checpoint()
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                metrics['ProcessID'] = os.getpid()
                logger.log_metrics_summary(metrics, default_stats='mean')

            self.step += 1

    def save_checpoint(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        if self.return_type == 2 or self.return_type == 3 or self.return_type == 5 or self.return_type == 6:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'reward_model': self.step_reward_prediction_model.state_dict(),
            }
        elif self.return_type == 9:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'state_predict_model': self.state_predict_model.state_dict(),
                'ema_state_predict_model': self.ema_state_predict_model.state_dict(),
            }
        else:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict()
            }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        savepath = os.path.join(savepath, f'state_{self.step}.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        if self.return_type == 2 or self.return_type == 3 or self.return_type == 5 or self.return_type == 6:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'reward_model': self.step_reward_prediction_model.state_dict(),
            }
        elif self.return_type == 9:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
                'state_predict_model': self.state_predict_model.state_dict(),
                'ema_state_predict_model': self.ema_state_predict_model.state_dict(),
            }
        else:
            data = {
                'step': self.step,
                # 'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict()
            }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        # self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if self.return_type == 2 or self.return_type == 3 or self.return_type == 5 or self.return_type == 6:
            self.step_reward_prediction_model.load_state_dict(data['reward_model'])
        elif self.return_type == 9:
            self.state_predict_model.load_state_dict(data['state_predict_model'])
            self.ema_state_predict_model.load_state_dict(data['ema_state_predict_model'])
    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                if self.return_type == 0 or self.return_type == 1:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 2 or self.return_type == 3:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 9:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 4:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 5:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 6:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 7:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 8:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                else:
                    raise Exception("Config.return_type is set wrong value.")
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                if self.return_type == 0 or self.return_type == 1:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 2 or self.return_type == 3:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1)*0.9, torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 9:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1)*0.9, torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 4:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 5:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 6:
                    returns = to_device(torch.ones(n_samples, 1), self.device)
                elif self.return_type == 7:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                elif self.return_type == 8:
                    returns = to_device(torch.cat([torch.ones(n_samples, 1)*0.9, torch.ones(n_samples, 1), torch.zeros(n_samples, 1)], dim=-1), self.device)
                else:
                    raise Exception("Config.return_type is set wrong value.")
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                if self.return_type == 9:
                    samples = self.ema_state_predict_model.conditional_sample(conditions, returns=returns)
                else:
                    samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)
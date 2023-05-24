import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
import time
import datetime
from datetime import datetime

def add_noise(obs, noise_scale=0.1):
    if isinstance(obs, list):
        for i in range(len(obs)):
            obs[i] += np.random.normal(loc=0.0, scale=noise_scale, size=len(obs[i]))
    if isinstance(obs, np.ndarray) and len(np.shape(obs)) == 1:
        obs += np.random.normal(loc=0.0, scale=noise_scale, size=len(obs))
    if isinstance(obs, np.ndarray) and len(np.shape(obs)) == 2:
        for i in range(len(obs)):
            obs[i] += np.random.normal(loc=0.0, scale=noise_scale, size=len(obs[i]))
    return obs

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{Config.eval_step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)
    logger.print(f"current_model_train_step:{state_dict['step']}", color='green')

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        include_goal_returns=Config.include_goal_returns,
        goal_distance=Config.goal_distance,
        return_type=Config.return_type,
        traj_length_must_bigger_than=Config.traj_length_must_bigger_than,
        top_K_length=Config.top_K_length,
        traj_return_must_bigger_than=Config.traj_return_must_bigger_than,
        history_length=Config.history_length,
        distance_to_failure_obs=Config.distance_to_failure_obs,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    assert Config.diffusion_goal_prediction + Config.general_goal_prediction == 1

    if Config.general_goal_prediction:
        return_based_state_to_goal_config = utils.Config(
            Config.return_based_state_to_goal,
            savepath='return_based_state_to_goal.pkl',
            state_dim=observation_dim,
            goal_dim=observation_dim,
            state_to_goal_lr=3e-4,
            device=Config.device
        )

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        if Config.return_type == 4:
            model_config = utils.Config(
                Config.transformer_model,
                savepath='model_config.pkl',
                n_layers=1,
                sequen_dim=observation_dim,
                calc_energy=Config.calc_energy,
                n_heads=8,
                dim=Config.transformer_hidden_dim,
                max_seq_len=1000,
                slide_seq_len=Config.horizon,
                return_type=Config.return_type,
                condition_dropout=Config.condition_dropout,
                goals_condition=Config.goals_condition,
                returns_condition=Config.returns_condition,
            )
        elif Config.return_type == 7:
            model_config = utils.Config(
                Config.model,
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim + 1,
                cond_dim=observation_dim + 1,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device,
                goals_condition=Config.goals_condition,
                goal_dim=observation_dim,
                return_type=Config.return_type,
            )
        else:
            model_config = utils.Config(
                Config.model,
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=transition_dim,
                cond_dim=observation_dim,
                dim_mults=Config.dim_mults,
                dim=Config.dim,
                returns_condition=Config.returns_condition,
                device=Config.device,
                goals_condition=Config.goals_condition,
                goal_dim=observation_dim,
                return_type=Config.return_type,
            )

        if Config.goals_condition:
            if Config.general_goal_prediction:
                return_based_s2g_model = return_based_state_to_goal_config()
            else:
                return_based_s2g_model = None
            # bisimulation_model = bisimulation_model_config()
            diffusion_config = utils.Config(
                Config.diffusion,
                savepath='diffusion_config.pkl',
                horizon=Config.horizon,
                observation_dim=observation_dim,
                action_dim=action_dim,
                n_timesteps=Config.n_diffusion_steps,
                loss_type=Config.loss_type,
                clip_denoised=Config.clip_denoised,
                predict_epsilon=Config.predict_epsilon,
                hidden_dim=Config.hidden_dim,
                ar_inv=Config.ar_inv,
                train_only_inv=Config.train_only_inv,
                ## loss weighting
                action_weight=Config.action_weight,
                loss_weights=Config.loss_weights,
                loss_discount=Config.loss_discount,
                returns_condition=Config.returns_condition,
                condition_guidance_w=Config.condition_guidance_w,
                device=Config.device,
                goals_condition=Config.goals_condition,
                bisimulation_model=None,
                return_based_s2g_model=return_based_s2g_model,
                goal_distance=Config.goal_distance,
                general_goal_prediction=Config.general_goal_prediction,
                diffusion_goal_prediction=Config.diffusion_goal_prediction,
                return_type=Config.return_type,
            )
        else:
            diffusion_config = utils.Config(
                Config.diffusion,
                savepath='diffusion_config.pkl',
                horizon=Config.horizon,
                observation_dim=observation_dim,
                action_dim=action_dim,
                n_timesteps=Config.n_diffusion_steps,
                loss_type=Config.loss_type,
                clip_denoised=Config.clip_denoised,
                predict_epsilon=Config.predict_epsilon,
                hidden_dim=Config.hidden_dim,
                ## loss weighting
                action_weight=Config.action_weight,
                loss_weights=Config.loss_weights,
                loss_discount=Config.loss_discount,
                returns_condition=Config.returns_condition,
                device=Config.device,
                condition_guidance_w=Config.condition_guidance_w,
                goals_condition=Config.goals_condition,
                general_goal_prediction=Config.general_goal_prediction,
                diffusion_goal_prediction=Config.diffusion_goal_prediction,
                return_type=Config.return_type,
                normed_zero_reward_value=dataset.normed_zero_reward_value,
                srs_inv_model=Config.srs_inv_model,
            )
    else:
        raise Exception("Config.diffusion is wrong !!!")


    if Config.return_type == 2:
        step_reward_prediction_model_config = utils.Config(
            Config.reward_linear_regression_model,
            savepath='reward_linear_regression_model.pkl',
            state_dim=observation_dim,
            action_dim=action_dim,
            regression_lr=3e-4,
            device=Config.device
        )
        step_reward_prediction_model = step_reward_prediction_model_config()
    elif Config.return_type == 3:
        step_reward_prediction_model_config = utils.Config(
            Config.reward_quantile_regression_model,
            savepath='reward_quantile_regression_model.pkl',
            state_dim=observation_dim,
            action_dim=action_dim,
            quantile_lr=3e-4,
            device=Config.device
        )
        step_reward_prediction_model = step_reward_prediction_model_config()
    elif Config.return_type == 5 or Config.return_type == 6:
        if Config.return_type == 5:
            v_max = (1 - dataset.discount ** (dataset.max_path_length - 1)) / (1 - dataset.discount) * 1.5
            v_min = 0
        elif Config.return_type == 6:
            v_max = dataset.max_discounted_trajectory_return * 1.5
            v_min = dataset.min_discounted_trajectory_return-np.abs(dataset.min_discounted_trajectory_return)*0.5
        else:
            raise Exception("Config.return_type is wrong !!!")
        step_reward_prediction_model_config = utils.Config(
            Config.q_distribution_model,
            savepath='q_distribution_model.pkl',
            state_dim=observation_dim,
            action_dim=action_dim,
            max_action=float(dataset.env.action_space.high[0]),
            num_atoms=201,
            v_min=v_min,
            v_max=v_max,
            discount=Config.discount,
            ema_decay=Config.ema_decay,
            actor_lr=1e-4,
            critic_lr=1e-4,
            target_network_update_freq=10,
            device=Config.device
        )
        step_reward_prediction_model = step_reward_prediction_model_config()
    else:
        step_reward_prediction_model = None

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        general_goal_prediction=Config.general_goal_prediction,
        diffusion_goal_prediction=Config.diffusion_goal_prediction,
        return_type=Config.return_type,
        wandb_log=Config.wandb_log,
        wandb_log_frequency=Config.wandb_log_frequency,
        wandb_project_name=Config.wandb_project_name,
        step_reward_prediction_model=step_reward_prediction_model,
        history_length=Config.history_length,
    )

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    # todo  ================================  reload the saved model start  ===================================================
    trainer.step = state_dict['step']
    # trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])
    if Config.return_type == 2 or Config.return_type == 3 or Config.return_type == 5 or Config.return_type == 6:
        trainer.step_reward_prediction_model.load_state_dict(state_dict['reward_model'])
    logger.print(f"current_model_train_step:{state_dict['step']}, reward_return_scale:{dataset.reward_return_scale}", color='green')
    # todo ================================  reload the saved model end   ===================================================

    num_eval = 30
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    episode_rewards_sequence = [[] for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    # if Config.return_type == 0:
    #     Config.test_ret = dataset.normed_max_discounted_return# * Config.test_ret
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)
    if Config.return_type == 6:
        returns = to_device(dataset.max_discounted_trajectory_return / dataset.reward_return_scale * torch.ones(num_eval, 1), device)

    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    init_return_to_go = 1.0 * torch.ones(num_eval, 1)
    normal_return_to_go_mode = False
    if normal_return_to_go_mode:
        actionreward_return_to_go = 0.65 * torch.ones(num_eval, 1)
        every_step_normed_reward_cond = 0.9 * torch.ones(num_eval, 1)
    else:
        every_step_normed_reward_cond = (dataset.best_traj_step_rewards - dataset.min_step_rewards) / (dataset.max_step_rewards - dataset.min_step_rewards)
        if Config.return_type == 8:  # todo first_action_reward_return_to_go unify_max_traj_reward unity_max_single_step_reward
            every_step_normed_reward_cond = (dataset.best_traj_step_rewards - dataset.min_single_step_reward) / (dataset.max_single_step_reward - dataset.min_single_step_reward)
        actionreward_return_to_go = (every_step_normed_reward_cond.mean()) * torch.ones(num_eval, 1)
    current_num_eval_rewards = [0 for _ in range(num_eval)]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]

    while sum(dones) <  num_eval:
        step_start_time = time.time()
        obs = dataset.normalizer.normalize(obs, 'observations')
        # obs = add_noise(obs=obs)
        conditions = {0: to_torch(obs, device=device)}
        if Config.goals_condition:
            samples = trainer.ema_model.conditional_sample(conditions, goals=None, timestep=t)
        else:
            if Config.return_type == 2:
                return_to_go = (dataset.max_trajectory_return + Config.max_trajectory_return_offset - np.reshape(
                    np.array(episode_rewards), [num_eval, 1]) - dataset.min_trajectory_return) / (dataset.max_trajectory_return + Config.max_trajectory_return_offset - dataset.min_trajectory_return)
                return_to_go = torch.tensor(return_to_go)
                reward_prediction_input = to_device(torch.cat([torch.tensor(obs), return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
                first_action_reward_cond = trainer.step_reward_prediction_model.get_value(state=reward_prediction_input[:, :-2], return_to_go=reward_prediction_input[:, -2], timestep=reward_prediction_input[:, -1])
                returns = to_device(torch.cat([first_action_reward_cond.cpu(), return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
            if Config.return_type == 3:
                return_to_go = (dataset.max_trajectory_return + Config.max_trajectory_return_offset - np.reshape(
                    np.array(episode_rewards), [num_eval, 1]) - dataset.min_trajectory_return) / (dataset.max_trajectory_return + Config.max_trajectory_return_offset - dataset.min_trajectory_return)
                return_to_go = torch.tensor(return_to_go)
                reward_prediction_input = to_device(torch.cat([torch.tensor(obs), return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
                first_action_reward_cond = trainer.step_reward_prediction_model.get_quantile_points(state=reward_prediction_input[:, :-2], return_to_go=reward_prediction_input[:, -2], timestep=reward_prediction_input[:, -1])
                returns = to_device(torch.cat([first_action_reward_cond[:, 5:6].cpu(), return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
            if Config.return_type == 4:
                if t == 0:
                    return_to_go = init_return_to_go
                else:
                    return_to_go = (dataset.max_trajectory_return - np.reshape(np.array(episode_rewards), [num_eval, 1])  - dataset.min_trajectory_return)/(dataset.max_trajectory_return - dataset.min_trajectory_return)
                    return_to_go = torch.tensor(return_to_go)
                returns = to_device(torch.cat([return_to_go, t/dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
            if Config.return_type == 5:
                z_pmf = trainer.step_reward_prediction_model.critic(to_torch(obs, device=device), trainer.step_reward_prediction_model.actor(to_torch(obs, device=device)))
                mean_q = torch.sum(z_pmf * to_torch(trainer.step_reward_prediction_model.target_z, device=device), dim=-1, keepdim=True)
                returns = mean_q / dataset.hard_reward_return_scale
            if Config.return_type == 6:
                z_pmf = trainer.step_reward_prediction_model.critic(to_torch(obs, device=device), trainer.step_reward_prediction_model.actor(to_torch(obs, device=device)))
                mean_q = torch.sum(z_pmf * to_torch(trainer.step_reward_prediction_model.target_z, device=device), dim=-1, keepdim=True)
                returns = mean_q / dataset.reward_return_scale
            if Config.return_type == 7:
                if t == 0:
                    return_to_go = init_return_to_go
                    padding_rds = np.zeros((num_eval, Config.history_length, 1))
                    action_reward_seq = padding_rds
                else:
                    return_to_go = (dataset.max_trajectory_return + Config.max_trajectory_return_offset - np.reshape(np.array(episode_rewards), [num_eval, 1]) - dataset.min_trajectory_return) / (dataset.max_trajectory_return - dataset.min_trajectory_return)
                    return_to_go = torch.tensor(return_to_go)
                    action_reward_seq = np.reshape(np.array(episode_rewards_sequence), [num_eval, -1, 1])
                    action_reward_seq = action_reward_seq[:, -Config.history_length + 1:, :]
                    if np.shape(action_reward_seq)[1] < Config.history_length - 1:
                        head_padding_rds = np.zeros((num_eval, Config.history_length - 1 - np.shape(action_reward_seq)[1], 1))
                        tail_padding_rds = np.zeros((num_eval, 1, 1))
                        action_reward_seq = np.concatenate([head_padding_rds, action_reward_seq, tail_padding_rds], axis=1)
                    else:
                        tail_padding_rds = np.zeros((num_eval, 1, 1))
                        action_reward_seq = np.concatenate([action_reward_seq, tail_padding_rds], axis=1)

                returns = to_device(torch.cat([return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
                returns = torch.unsqueeze(returns, dim=1)
                returns = torch.tile(returns, (1, Config.sample_trajectory, 1))
                returns = torch.reshape(returns, [num_eval * Config.sample_trajectory, -1])
                conditions_obs_sequence = np.concatenate(recorded_obs[-Config.history_length:], axis=1)
                if len(recorded_obs) < Config.history_length:
                    padding_obs = np.zeros((num_eval, Config.history_length - len(recorded_obs), dataset.observation_dim))
                    conditions_obs_sequence = np.concatenate([padding_obs, conditions_obs_sequence], axis=1)
                conditions_obs_sequence = dataset.normalizer.normalize(conditions_obs_sequence, 'observations')
                action_reward_seq = (action_reward_seq - dataset.min_single_step_reward) / (dataset.max_single_step_reward - dataset.min_single_step_reward)
                diffusion_model_condition_input = np.expand_dims(
                    np.concatenate([conditions_obs_sequence, action_reward_seq], axis=-1), axis=1)
                diffusion_model_condition_input = np.tile(diffusion_model_condition_input, (1, Config.sample_trajectory, 1, 1))
                diffusion_model_condition_input = np.reshape(diffusion_model_condition_input, [num_eval * Config.sample_trajectory, Config.history_length, dataset.observation_dim + 1])
                conditions = {Config.history_length - 1: to_torch(diffusion_model_condition_input, device=device)}
            if Config.return_type == 8:
                if normal_return_to_go_mode:
                    first_action_reward_cond = 0.93 * torch.ones(num_eval, 1)
                else:
                    try:
                        first_action_reward_cond = every_step_normed_reward_cond[t+Config.history_length-1] * torch.ones(num_eval, 1)
                    except:
                        first_action_reward_cond = 0. * torch.ones(num_eval, 1)
                return_to_go = (dataset.max_trajectory_return + Config.max_trajectory_return_offset - np.reshape(np.array(episode_rewards), [num_eval, 1]) - dataset.min_trajectory_return) / (dataset.max_trajectory_return - dataset.min_trajectory_return)
                return_to_go = torch.tensor(return_to_go)
                returns = to_device(torch.cat([first_action_reward_cond, return_to_go, t / dataset.max_path_length * torch.ones(num_eval, 1)], dim=-1), device, convert_to_torch_float=True)
                conditions_obs_sequence = np.concatenate(recorded_obs[-Config.history_length:], axis=1)
                if len(recorded_obs) < Config.history_length:
                    padding_obs = np.zeros((num_eval, Config.history_length - len(recorded_obs), dataset.observation_dim))
                    conditions_obs_sequence = np.concatenate([padding_obs, conditions_obs_sequence], axis=1)
                conditions_obs_sequence = dataset.normalizer.normalize(conditions_obs_sequence, 'observations')
                conditions = {Config.history_length - 1: to_torch(conditions_obs_sequence, device=device)}


            samples = trainer.ema_model.conditional_sample(conditions, returns=returns)


        if Config.return_type == 8:
            obs_comb = torch.cat([samples[:, Config.history_length-1, :], samples[:, Config.history_length, :]], dim=-1)
        elif Config.return_type == 7:
            samples = torch.reshape(samples, [num_eval, Config.sample_trajectory, Config.horizon, dataset.observation_dim + 1])
            sorted_index = samples[:, :, Config.history_length - 1, -1:]
            # sorted_index = torch.sum(samples[:, :, Config.history_length - 1:, -1:], dim=-2)
            sorted_index = torch.argsort(sorted_index, dim=1, descending=True)
            sorted_index = torch.reshape(sorted_index[:, 0, :], [num_eval, 1, 1, 1])
            samples = torch.squeeze(torch.gather(samples, dim=1, index=torch.tile(sorted_index, [1, 1, Config.horizon, dataset.observation_dim + 1])), dim=1)
            obs_comb = torch.cat([samples[:, Config.history_length - 1, :], samples[:, Config.history_length, :-1]], dim=-1) if Config.srs_inv_model else torch.cat([samples[:, Config.history_length - 1, :-1], samples[:, Config.history_length, :-1]], dim=-1)

        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        if Config.return_type == 7:
            obs_comb = obs_comb.reshape(-1, 2 * observation_dim+1) if Config.srs_inv_model else obs_comb.reshape(-1, 2 * observation_dim)
        else:
            obs_comb = obs_comb.reshape(-1, 2 * observation_dim)

        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        if t == 0:
            if Config.return_type == 7:
                normed_observations = samples[:, :, :-1]
            else:
                normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            # savepath = os.path.join('images', 'sample-planned.png')
            # renderer.composite(savepath, observations, plt_fig_path="plt_images")

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            current_num_eval_rewards[i] = this_reward
            if this_done:
                if dones[i] == 1:
                    episode_rewards_sequence[i].append(0)
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    episode_rewards_sequence[i].append(this_reward)
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    episode_rewards_sequence[i].append(0)
                    pass
                else:
                    episode_rewards[i] += this_reward
                    episode_rewards_sequence[i].append(this_reward)
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1
        print(f"eval_process>>>>> env_timestep:{t}, return_range:[{torch.min(returns[:, 0]).detach().cpu().numpy(), torch.max(returns[:, 0]).detach().cpu().numpy()}], current_reward:{episode_rewards}, time_consumption:{time.time()-step_start_time}, normed_ep_return:{actionreward_return_to_go.mean()}")

    episode_rewards = np.array(episode_rewards)

    logger.print(f"current_model_train_step: {state_dict['step']}, max_trajectory_return_offset: {Config.max_trajectory_return_offset}, return_type: {Config.return_type}, "
                 f"history_length: {Config.history_length}, distance_to_failure_obs: {Config.distance_to_failure_obs}, reward_return_scale: {dataset.reward_return_scale}, "
                 f"top_K: {Config.top_K_length}", color='green')
    logger.print(f"bucket:{Config.bucket} env:{Config.dataset} Config.test_ret:{Config.test_ret}")
    logger.print(f"ep_reward: {episode_rewards}, normed_score:{gym.make(Config.dataset).get_normalized_score(episode_rewards)}", color='green')
    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward': np.mean(episode_rewards), 'std_ep_reward': np.std(episode_rewards)})

    if dataset.dataset_name.split("-")[0] not in ["hammer", "pen", "relocate", "door"]:
        recorded_obs = np.concatenate(recorded_obs, axis=1)
        now_time = datetime.now()
        savepath = os.path.join('images', f'sample-executed-{now_time.year}{now_time.month}{now_time.day}.png')
        renderer.composite(savepath, recorded_obs, plt_fig_path="plt_images")


import diffuser.utils as utils
import torch
import numpy as np

def main(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)
    father_save_path = Config.bucket

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    if Config.reset_seed:
        Config.seed = np.random.randint(0, 999)
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        father_save_path=father_save_path,
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
        father_save_path=father_save_path,
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config() if Config.dataset.split("-")[0] != "temporal_condition" else None
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ goal generation ------------------------------#
    # -----------------------------------------------------------------------------#
    assert Config.diffusion_goal_prediction + Config.general_goal_prediction == 1
    if Config.general_goal_prediction:
        return_based_state_to_goal_config = utils.Config(
            Config.return_based_state_to_goal,
            savepath='return_based_state_to_goal.pkl',
            father_save_path=father_save_path,
            state_dim=observation_dim,
            goal_dim=observation_dim,
            state_to_goal_lr=3e-4,
            device=Config.device
        )
    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        if Config.return_type == 4:
            model_config = utils.Config(
                Config.transformer_model,
                savepath='model_config.pkl',
                father_save_path=father_save_path,
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
                father_save_path=father_save_path,
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
                father_save_path=father_save_path,
                horizon=Config.horizon,
                transition_dim=observation_dim,
                cond_dim=observation_dim,
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

        if Config.goals_condition:
            if Config.general_goal_prediction:
                return_based_s2g_model =  return_based_state_to_goal_config()
            else:
                return_based_s2g_model = None
            diffusion_config = utils.Config(
                Config.diffusion,
                savepath='diffusion_config.pkl',
                father_save_path=father_save_path,
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
                father_save_path=father_save_path,
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
                general_goal_prediction=Config.general_goal_prediction,
                diffusion_goal_prediction=Config.diffusion_goal_prediction,
                return_type=Config.return_type,
                normed_zero_reward_value=dataset.normed_zero_reward_value,
                srs_inv_model=Config.srs_inv_model,
            )
    else:
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            father_save_path=father_save_path,
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            father_save_path=father_save_path,
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )

    if Config.return_type == 2:
        step_reward_prediction_model_config = utils.Config(
            Config.reward_linear_regression_model,
            savepath='reward_linear_regression_model.pkl',
            father_save_path=father_save_path,
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
            father_save_path=father_save_path,
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
            v_max = dataset.max_discounted_trajectory_return * 2.0
            v_min = dataset.min_discounted_trajectory_return-np.abs(dataset.min_discounted_trajectory_return)*0.5
        else:
            raise Exception("Config.return_type is wrong !!!")
        step_reward_prediction_model_config = utils.Config(
            Config.q_distribution_model,
            savepath='q_distribution_model.pkl',
            father_save_path=father_save_path,
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
        father_save_path=father_save_path,
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
        save_checkpoints=Config.save_checkpoints,
        general_goal_prediction=Config.general_goal_prediction,
        diffusion_goal_prediction=Config.diffusion_goal_prediction,
        return_type=Config.return_type,
        wandb_log=Config.wandb_log,
        wandb_log_frequency=Config.wandb_log_frequency,
        wandb_project_name=Config.wandb_project_name,
        step_reward_prediction_model=step_reward_prediction_model,
        history_length=Config.history_length,
        save_range=Config.save_range
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    if Config.return_type == 5 or Config.return_type == 6:
        loss, _ = diffusion.loss(x=batch.trajectories, cond=batch.conditions, returns=batch.returns)
        loss.backward()
    elif Config.return_type == 7:
        batch = type(batch)(torch.cat([batch.trajectories, batch.FARSequence], dim=-1), batch.conditions, batch.returns_timesteps, batch.FARSequence)
        batch.conditions.clear()
        batch.conditions.update({Config.history_length - 1: batch.trajectories[:, 0:Config.history_length, dataset.action_dim:]})
        loss, infos = diffusion.loss(x=batch.trajectories, cond=batch.conditions, returns=batch.returns_timesteps)
        loss.backward()
    else:
        loss, _ = diffusion.loss(*batch)
        loss.backward()
    logger.print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch, n_epochs=i)


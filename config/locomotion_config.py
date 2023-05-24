import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


# todo           return_type = [0,  1,                   2,      3,       4,   5,               6,   7,   8, ]
# todo  corresponding method = [DD, DD with hard reward, RR-TCD, RQR-TCD, TFD, DQD hard reward, DQD, SRD, TCD,]

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = 'analysis/code/model_data/walker2d_e_RT8/'
    dataset = 'walker2d-expert-v2'
    # todo       halfcheetah    hopper     walker2d               random   medium  expert   medium-expert   medium-replay  full-replay                  v2
    # todo       maze2d                                           umaze   medium    large    umaze-dense   medium-dense   large-dense        v1
    # todo       pen    relocate                                  human   expert    cloned                                                   v1

    ## model
    model = 'models.TemporalUnet'
    transformer_model = 'models.Transformer'
    behavior_cloning_model = 'models.BC'
    diffusion = 'models.GaussianInvDynDiffusion'
    behavior_cloning_diffusion = 'models.BehaviorCloningDiffusion'
    horizon = 100
    n_diffusion_steps = 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=1.0
    condition_guidance_w = 1.2
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    # hard_reward = False
    return_type = 8
    traj_length_must_bigger_than = 100
    traj_return_must_bigger_than = None
    top_K_length = 1
    max_trajectory_return_offset=0
    history_length = 5
    distance_to_failure_obs = 0
    transformer_hidden_dim = 256
    sample_trajectory = 3
    srs_inv_model = False
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = 1000000
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 10000
    save_range = [100000, 900000]
    eval_step = 100000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = False

    ## goal_conditional_generation
    bisimulation_model = 'models.Bisimulation_Model'
    return_based_state_to_goal = 'models.Return_based_State_to_Goal_Model'
    goal_conditional_generation = False
    goal_distance = 1
    include_goal_returns=False
    goals_condition = False
    general_goal_prediction = False
    diffusion_goal_prediction = True

    # wandb log config
    wandb_project_name = 'diffusion_offline'
    wandb_log = False
    wandb_log_frequency = 1000
    reset_seed = False

    # reward prediction model
    reward_linear_regression_model = 'models.Linear_Regression_Model'
    reward_quantile_regression_model = 'models.Quantile_Regression_Model'
    q_distribution_model = 'models.Categorical_Q_Function'

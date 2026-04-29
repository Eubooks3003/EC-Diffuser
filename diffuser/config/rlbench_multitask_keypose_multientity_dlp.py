"""2D multi-task keypose v2 config -- MULTIENTITY flavor.

Same data and view setup as the multiview multitask config (front + overhead,
2 views x 20 kp_enc = 40 particles), but with split_action_tokens=True so the
denoiser decodes the action as three separate sub-tokens (pos, rot, grip)
instead of a single monolithic token.

See rlbench_multitask_keypose_multiview_dlp.py for the full template/notes;
only the multientity-specific knobs are commented here.
"""

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

DATA_ROOT = '/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench/preprocessed_multiview_tokens_with_keyposes'

TASK_NAMES = [
    'close_jar',
    'meat_off_grill',
    'open_drawer',
    'push_buttons',
    'put_item_in_drawer',
    'reach_and_drag',
    'slide_block_to_color_target',
    'stack_blocks',
    'sweep_to_dustpan_of_size',
    'turn_tap',
]

TASK_PKLS      = [f'{DATA_ROOT}/rlbench_{t}/rlbench_{t}.pkl'  for t in TASK_NAMES]
TASK_DLP_CKPTS = [f'{DATA_ROOT}/rlbench_{t}/dlp_ckpt.pt'      for t in TASK_NAMES]
TASK_DLP_CFGS  = [f'{DATA_ROOT}/rlbench_{t}/dlp_config.json'  for t in TASK_NAMES]

_REP_TASK_IDX = TASK_NAMES.index('meat_off_grill')

mode_to_args = {
  '16C_dlp': {
    'keypose_mode': True,

    'dataset': 'multitask',
    'override_dataset_path': TASK_PKLS,
    'calib_h5_path': None,

    'dlp_ckpt': TASK_DLP_CKPTS[_REP_TASK_IDX],
    'dlp_ctor': "models:DLP",
    'dlp_cfg': TASK_DLP_CFGS[_REP_TASK_IDX],

    'task_names':     TASK_NAMES,
    'task_dlp_ckpts': TASK_DLP_CKPTS,
    'task_dlp_cfgs':  TASK_DLP_CFGS,

    'features_dim': 10,
    'gripper_dim': 10,
    'use_gripper_obs': True,  # cond-tokens path: current gripper feeds into cond[0]
    'gripper_state_mask_ratio': 0.0,
    'split_action_tokens': True,  # multientity: action decoded as pos/rot/grip sub-tokens
    'bg_dim': 8,              # 2 views x learned_bg_feature_dim(4)
    'use_bg_obs': True,
    'max_particles': 40,      # 2 views x n_kp_enc=20
    'multiview': True,
    'use_views': [0, 1],      # 0=front, 1=overhead
    'num_source_views': 4,
    'device': 'cuda:0',
    'max_path_length': 1000,
    'max_demos': 100,
    'eval_freq': 60,
    'eval_backend': 'none',
    'rlbench_eval_episodes': 2,
    'rlbench_eval_max_steps': 20,
    'rlbench_eval_cams': ['front', 'overhead', 'left_shoulder', 'right_shoulder'],
    'rlbench_eval_image_size': 128,
    'rlbench_eval_video_fps': 10,
    'n_steps_per_epoch': 500,
    'action_dim': 10,
    'act_rot_dim': 6,
    'lang_dim': 512,
    'lang_pooled': False,
    'max_lang_tokens': 32,
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'lang_device': 'cpu',
    "use_absolute_actions": True,
    'horizon': 1,
    'exe_steps': 1,
    "random_init": True,
    "random_init_eval": True,
  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'keypose_mode': True,
        'horizon': 1,
        'features_dim': 10,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 12,
        'dropout': 0.0,

        'n_diffusion_steps': 100,
        'action_weight': 1,

        'max_particles': 40,
        'positional_bias': False,
        'multiview': True,

        'loader': 'datasets.LanguageConditionedDataset',
        'normalizer': 'GaussianNormalizer',
        'particle_normalizer': 'ParticleGaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 10,
        'obs_only': False,
        'action_only': False,
        'action_z_scale': 1.0,
        'gripper_state_mask_ratio': 0.0,

        'logbase': logbase,
        'prefix': 'diffusion/rlbench_multitask_keypose_multientity/',
        'exp_name': watch(args_to_watch),

        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,
        'batch_size': 128,
        'learning_rate': 8e-5,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 10_000,
        'eval_freq': 10**9,
        'sample_freq': 1,
        'n_saves': 2,
        'save_parallel': False,
        'n_reference': 1,
        'bucket': None,
        'device': 'cuda:0',
        'seed': 0,
        'renderer': 'utils.ParticleRenderer3D',
        'predict_epsilon': False,
        'env_config_dir': 'env_config/n_cubes',

        'loss_weights': None,
        'loss_discount': 1,

        'exe_steps': 1,
    },

    'plan': {
        'policy': 'sampling.LanguageConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 1,

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/rlbench_multitask_keypose_multientity/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        'diffusion_epoch': 'latest',
        'horizon': 2,
        'n_diffusion_steps': 5,
        'verbose': False,
        'suffix': 'f:step_{diffusion_epoch}',
    },
}

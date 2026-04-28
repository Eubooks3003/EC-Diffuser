"""Multi-task keypose v2 config across all 10 RLBench-PerAct tasks.

This config drives a single keypose-mode policy on all tasks jointly (same setup
as 3DDA / RVT / PerAct headline numbers). Language conditioning disambiguates
which task per episode.

NOTE: Requires a multi-pkl loader extension in `buffer.py` /
`SequenceDataset.__init__`. The current loader (`load_paths_from_pickle`)
accepts a single pkl path; multi-task requires concatenating episode-axis
across the list of per-task pkls with padding to global (max_path_length,
max_keyposes). See override_dataset_path below for the list of pkls to
concatenate.
"""

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# All 10 PerAct keypose tasks. Each pkl has shape (100 episodes, max_path_per_task, ...).
# Global max_path = 927 (stack_blocks); global max_keyposes = 24 (stack_blocks).
TASK_PKLS = [
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_close_jar/close_jar.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_meat_off_grill/meat_off_grill.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_open_drawer/open_drawer.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_push_buttons/push_buttons.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_put_item_in_drawer/put_item_in_drawer.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_reach_and_drag/reach_and_drag.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_slide_block_to_color_target/slide_block_to_color_target.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_stack_blocks/stack_blocks.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_sweep_to_dustpan_of_size/sweep_to_dustpan_of_size.pkl',
    '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_turn_tap/turn_tap.pkl',
]

# RLBench multitask language-conditioned config (keypose v2).
# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
mode_to_args = {
  '16C_dlp': {
    'keypose_mode': True,

    # Multi-task: list of pkls to concatenate. The DLP encoder is shared across
    # all 10 tasks (single ckpt below) since it was trained jointly.
    'dataset': 'multitask',
    'override_dataset_path': TASK_PKLS,
    'calib_h5_path': None,
    'dlp_ckpt': '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_meat_off_grill/dlp_ckpt.pt',
    'dlp_ctor': "voxel_models:DLP",
    'dlp_cfg': '/home/ellina/Desktop/data/preprocessed_voxel_tokens_with_keyposes/rlbench_meat_off_grill/dlp_config.json',
    'features_dim': 12,       # z(3)+scale(3)+depth(1)+obj_on(1)+feat(4)
    'gripper_dim': 10,        # pos(3)+rot6d(6)+open(1)
    'use_gripper_obs': True,  # cond-tokens path: current gripper feeds into cond[0]
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 2,
    'use_bg_obs': True,
    'max_particles': 40,
    'multiview': False,
    'device': 'cuda:0',
    'max_path_length': 927,   # global max across the 10 tasks (stack_blocks)
    'max_demos': 100,         # per-task; total = 10 * 100 = 1000 episodes
    'eval_freq': 60,
    'eval_backend': 'rlbench',
    # Per-task live eval — each task evaluated separately at eval time.
    'rlbench_eval_episodes': 2,
    'rlbench_eval_max_steps': 20,  # keypose attempts per episode
    'rlbench_eval_cams': ['front', 'overhead', 'left_shoulder', 'right_shoulder'],
    'rlbench_eval_image_size': 128,
    'rlbench_eval_video_fps': 10,
    'n_steps_per_epoch': 500,
    # --- RLBench-specific ---
    'action_dim': 10,
    'lang_dim': 512,
    'lang_pooled': False,
    'max_lang_tokens': 32,
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'lang_device': 'cpu',
    # -------------------------
    "use_absolute_actions": True,
    'horizon': 1,    # predict only the next keypose (3DDA-style); cond is current keypose
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
        'features_dim': 7,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 12,        # multi-task: data is plentiful, capacity helps
        'dropout': 0.0,

        'n_diffusion_steps': 100,
        'action_weight': 1,

        'max_particles': 64,
        'positional_bias': False,
        'multiview': False,

        # dataset
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

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/rlbench_multitask_keypose/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,
        'batch_size': 32,       # multi-task: 10x data, larger batch keeps tasks balanced per step
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
        'prefix': 'plans/rlbench_multitask_keypose/',
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

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# RLBench 2D-DLP language-conditioned config for task: sweep_to_dustpan_of_size
# Multiview + keypose + split action tokens (separate pos/rot/grip heads).
mode_to_args = {
  '16C_dlp': {
    'keypose_mode': True,

    'dataset': 'sweep_to_dustpan_of_size',
    'override_dataset_path': '/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench/preprocessed_multiview_tokens_with_keyposes/rlbench_sweep_to_dustpan_of_size/rlbench_sweep_to_dustpan_of_size.pkl',
    'calib_h5_path': None,
    'dlp_ckpt': '/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench/preprocessed_multiview_tokens_with_keyposes/rlbench_sweep_to_dustpan_of_size/dlp_ckpt.pt',
    'dlp_ctor': "models:DLP",
    'dlp_cfg': '/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench/preprocessed_multiview_tokens_with_keyposes/rlbench_sweep_to_dustpan_of_size/dlp_config.json',
    'features_dim': 10,
    'gripper_dim': 10,
    'use_gripper_obs': False,
    'split_action_tokens': True,  # multi-entity action: decode pos/rot/grip as 3 separate heads
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 8,              # 2 views x learned_bg_feature_dim(4)
    'use_bg_obs': True,
    'max_particles': 40,      # 2 views x n_kp_enc=20
    'multiview': True,
    'use_views': [0, 1],      # 0=front, 1=overhead
    'num_source_views': 4,
    'device': 'cuda:0',
    'max_path_length': 600,
    'max_demos': 100,
    'eval_freq': 10,
    'eval_backend': 'none',
    'n_steps_per_epoch': 500,
    # --- RLBench-specific ---
    'action_dim': 10,
    'act_rot_dim': 6,
    'lang_dim': 512,
    'lang_pooled': False,
    'max_lang_tokens': 10,
    'clip_model_name': 'openai/clip-vit-base-patch32',
    'lang_device': 'cpu',
    'rlbench_cams': ['front', 'overhead'],
    'rlbench_image_size': 128,
    'rlbench_headless': True,
    'rlbench_max_steps': 400,
    # -------------------------
    "use_absolute_actions": True,
    'horizon': 2,    # keypose chunk
    'exe_steps': 1,  # apply 1 keypose, replan
    "random_init": True,
    "random_init_eval": True,
    'save_gt_video': True,
    'demo_dataset_root': '/home/ubuntu/tal-lpwm-neurips-2026/data/rlbench/rlbench_rgb',
    'save_imagined': True,
    'save_imagined_recon': True,
  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'keypose_mode': True,
        'horizon': 2,
        'features_dim': 10,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.0,

        'n_diffusion_steps': 20,
        'action_weight': 1,

        'max_particles': 40,
        'positional_bias': False,
        'multiview': True,

        # dataset
        'loader': 'datasets.LanguageConditionedDataset',
        'normalizer': 'GaussianNormalizer',
        'particle_normalizer': 'ParticleGaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': False,
        'max_path_length': 10,
        'obs_only': False,
        'action_only': False,
        'action_z_scale': 1.0,
        'gripper_state_mask_ratio': 0.0,

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/rlbench_sweep_to_dustpan_of_size_keypose_multientity/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,
        'batch_size': 16,
        'learning_rate': 8e-5,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 10_000,
        'eval_freq': 10**9,
        'sample_freq': 1,
        'n_saves': 2,
        'save_parallel': False,
        'n_reference': 0,
        'bucket': None,
        'device': 'cuda:0',
        'seed': 0,
        'renderer': 'utils.ParticleRenderer',
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
        'prefix': 'plans/rlbench_sweep_to_dustpan_of_size_keypose_multientity/',
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

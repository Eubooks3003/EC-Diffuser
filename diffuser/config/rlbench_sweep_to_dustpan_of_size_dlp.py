from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# RLBench 2D-DLP language-conditioned config for task: sweep_to_dustpan_of_size
# Mode key matches setup.py: "{num_entity}C_{input_type}". Use --num_entity 16 --input_type dlp.
mode_to_args = {
  '16C_dlp': {
    'dataset': 'sweep_to_dustpan_of_size',
    'override_dataset_path': '/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_sweep_to_dustpan_of_size/rlbench_sweep_to_dustpan_of_size.pkl',
    'calib_h5_path': None,  # RLBench does not use a robomimic calib HDF5
    'dlp_ckpt': '/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_sweep_to_dustpan_of_size/dlp_ckpt.pt',
    'dlp_ctor': "models:DLP",
    'dlp_cfg': '/home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_sweep_to_dustpan_of_size/dlp_config.json',
    'features_dim': 10,       # Dtok from pkl meta (2D DLP multiview tokens: z2+scale2+depth1+obj_on1+feat4)
    'gripper_dim': 10,        # pos(3)+rot6d(6)+open(1)
    'use_gripper_obs': False,
    'gripper_state_mask_ratio': 0.0,
    'bg_dim': 8,              # 2 views x learned_bg_feature_dim(4)
    'use_bg_obs': True,
    'max_particles': 40,      # 2 views x n_kp_enc=20
    'multiview': True,
    # Slice multiview pkl down to front+overhead at load time:
    'use_views': [0, 1],      # 0=front, 1=overhead, 2=left_shoulder, 3=right_shoulder
    'num_source_views': 4,    # total views in the multiview pkl
    'device': 'cuda:0',
    'max_path_length': 600,   # Tmax from RLBench demos (pkl has 600 timesteps)
    'max_demos': 100,
    'eval_freq': 60,
    'eval_backend': 'none',
    'n_steps_per_epoch': 500,
    # --- RLBench-specific ---
    'action_dim': 10,                        # [pos(3), rot6d(6), open(1)] absolute EEF control
    'act_rot_dim': 6,             # rot6d occupies 6 action dims
    'lang_dim': 512,                          # CLIP ViT-B/32 hidden size
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
    'horizon': 6,
    'exe_steps': 3,
    "random_init": True,
    "random_init_eval": True,
    # Eval-time diagnostics (auto-exported to ECDIFF_* env vars by train.py /
    # eval_rlbench.py so no shell setup is needed).
    'save_gt_video': True,
    'demo_dataset_root': '/home/ellina/Desktop/data/rlbench_rgb',
    'save_imagined': True,
    'save_imagined_recon': True,
  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'horizon': 5,
        'features_dim': 10,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.0,

        'n_diffusion_steps': 5,
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
        'prefix': 'diffusion/rlbench_sweep_to_dustpan_of_size_multiview_fo/',
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
        'renderer': 'utils.ParticleRenderer',  # 2D renderer
        'predict_epsilon': False,
        'env_config_dir': 'env_config/n_cubes',

        'loss_weights': None,
        'loss_discount': 1,

        'exe_steps': 3,
    },

    'plan': {
        'policy': 'sampling.LanguageConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 3,

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/rlbench_sweep_to_dustpan_of_size_multiview_fo/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        'diffusion_epoch': 'latest',
        'horizon': 5,
        'n_diffusion_steps': 5,
        'verbose': False,
        'suffix': 'f:step_{diffusion_epoch}',
    },
}

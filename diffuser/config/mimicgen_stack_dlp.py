from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
# With your tokens: --num_entity 64 --input_type dlp  =>  "64C_dlp"
mode_to_args = {
  '16C_dlp': {
    'dataset': 'mimicgen_stack_dlp',
    'override_dataset_path': '/home/ellina/Desktop/Code/lpwm-dev/ecdiffuser_data/stack_replay_buffer_dlp.pkl',
    'calib_h5_path': '/home/ellina/Desktop/Code/articubot-on-mimicgen/stack_d1_rgbd_pcd.hdf5',
    'dlp_ckpt': '/home/ellina/Desktop/Code/lpwm-dev/checkpoints_3d/best/best.pt',
    'dlp_ctor': "voxel_models:DLP",
    'dlp_cfg': "/home/ellina/Desktop/Code/lpwm-dev/configs/best.json",
    'features_dim': 12,
    'max_particles': 16,
    'multiview': False,
    'device': 'cuda:0',
    'max_path_length': 132,
    'env_config_dir': 'env_config/n_cubes',
    'eval_freq': 20,
    'eval_backend': 'mimicgen',
    'n_steps_per_epoch': 1000,
    "mimicgen_cams": ["agentview", "sideview"],
    "mimicgen_camera_width": 256,
    "mimicgen_camera_height": 256,
    "mimicgen_max_steps":500

  },
}


base = {
    'diffusion': {
        'model': 'models.AdaLNPINTDenoiser',
        'diffusion': 'models.GaussianDiffusion',

        'horizon': 5,
        'features_dim': 7,
        'hidden_dim': 256,
        'projection_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.0,

        'n_diffusion_steps': 5,
        'action_weight': 10,

        'max_particles': 64,
        'positional_bias': False,
        'multiview': False,

        # dataset
        'loader': 'datasets.GoalDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'particle_normalizer': 'ParticleGaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 10,
        'obs_only': False,
        'action_only': False,

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/mimicgen_stack/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2e6,    
        'batch_size': 1,
        'learning_rate': 8e-5,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 200,
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

        # (safe to include; many configs rely on these existing)
        'loss_weights': None,
        'loss_discount': 1,

    },

    'plan': {
        # not used while eval_freq is huge
        'policy': 'sampling.GoalConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 1,

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/mimicgen_stack/',
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

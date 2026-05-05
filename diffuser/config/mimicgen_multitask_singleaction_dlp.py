"""
Multitask multiview mimicgen — single-action-token + split proprio.

Mirrors mimicgen_multitask_dlp.py but flips `split_action_tokens=False` so
pint.py emits one transformer token for the full action_dim=7 vector
(action_projection + action_encoding) while proprio stays split into
[p_pos, p_rot, p_grip] (gripper_dim=10, has_proprio=True). Token sequence:

    [action, p_pos, p_rot, p_grip, particle_1..K]

`split_action_tokens` is appended to args_to_watch so the experiment name
self-documents the flag, and `prefix` is bumped so savepaths never collide
with the multi-entity multitask runs.
"""

import os

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
    ('split_action_tokens', 'split'),
]

logbase = 'data'

TASK_NAMES = [
    'coffee',
    'coffee_preparation',
    'hammer_cleanup',
    'kitchen',
    'mug_cleanup',
    'nut_assembly',
    'pick_place',
    'square',
    'stack',
    'stack_three',
    'threading',
    'three_piece_assembly',
]
TASK_NAME_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}

TASK_MAX_STEPS = {
    'coffee':                600,
    'coffee_preparation':   1200,
    'hammer_cleanup':        700,
    'kitchen':              1000,
    'mug_cleanup':           700,
    'nut_assembly':          700,
    'pick_place':           1500,
    'square':                600,
    'stack':                 500,
    'stack_three':           600,
    'threading':            1000,
    'three_piece_assembly': 1000,
}


def _resolve_data_root():
    candidates = [
        '/lambda/nfs/tal-lpwm-neurips-2026/data/3D-DLP-mimicgen-data/preprocessed_multiview_tokens',
        '/home/ellina/Desktop/data/preprocessed_multiview_tokens',
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _resolve_calib_root():
    candidates = [
        '/lambda/nfs/tal-lpwm-neurips-2026/data/3D-DLP-mimicgen-data/core',
        '/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core',
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


DATA_ROOT = _resolve_data_root()
CALIB_ROOT = _resolve_calib_root()


def _build_task_entries():
    entries = []
    for name in TASK_NAMES:
        task_dir = f'{name}_d0'
        entries.append({
            'name': name,
            'task_id': TASK_NAME_TO_ID[name],
            'pkl': os.path.join(DATA_ROOT, task_dir, f'{task_dir}.pkl'),
            'calib_h5': os.path.join(CALIB_ROOT, f'{task_dir}.hdf5'),
            'dlp_ckpt': os.path.join(DATA_ROOT, task_dir, 'dlp_ckpt.pt'),
            'dlp_cfg':  os.path.join(DATA_ROOT, task_dir, 'dlp_config.json'),
            'max_steps': TASK_MAX_STEPS[name],
        })
    return entries


TASK_ENTRIES = _build_task_entries()


# IMPORTANT: key must match mode computed in setup.py: "{num_entity}C_{input_type}"
# Use --num_entity 12 to select this mode.
mode_to_args = {
    '12C_dlp': {
        'dataset': 'multitask',
        'multitask': True,
        'task_entries': TASK_ENTRIES,
        'task_names': TASK_NAMES,
        'n_tasks': len(TASK_NAMES),
        'max_demos_per_task': 200,

        'override_dataset_path': TASK_ENTRIES[0]['pkl'],
        'dlp_ckpt': TASK_ENTRIES[0]['dlp_ckpt'],
        'dlp_cfg':  TASK_ENTRIES[0]['dlp_cfg'],
        'dlp_ctor': 'models:DLP',
        'calib_h5_path': TASK_ENTRIES[0]['calib_h5'],

        # Shapes confirmed from per-task pkl meta (all 12 agree):
        # E=200, K=40 (20/view × 2 views), Dtok=10, A=7, G=10, BG=8 (4/view × 2 views)
        'features_dim': 10,
        'gripper_dim': 10,
        'use_gripper_obs': True,
        'gripper_state_mask_ratio': 0.0,
        'bg_dim': 8,
        'use_bg_obs': True,
        'max_particles': 48,
        'multiview': True,
        'device': 'cuda:0',

        'max_path_length': 800,

        'eval_freq': 0,
        'eval_backend': 'none',
        'n_steps_per_epoch': 500,

        'mimicgen_cams': ['agentview', 'sideview'],
        'mimicgen_camera_width': 256,
        'mimicgen_camera_height': 256,
        'mimicgen_max_steps': 600,
        'mimicgen_pixel_stride': 1,
        'use_absolute_actions': False,

        'horizon': 16,
        'exe_steps': 8,
        'random_init': True,
        'random_init_eval': True,

        # Single-action-token mode: pint.py uses one action_projection over
        # the full action_dim (7) instead of three (a_pos/a_rot/a_grip) heads.
        # Proprio stays split because gripper_dim > 0.
        'split_action_tokens': False,
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
        'action_weight': 50,

        'max_particles': 48,
        'positional_bias': False,
        'multiview': True,

        'multitask': False,
        'n_tasks': 1,

        'loader': 'datasets.MultitaskGoalDataset',
        'normalizer': 'GaussianNormalizer',
        'particle_normalizer': 'ParticleGaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 800,
        'obs_only': False,
        'action_only': False,
        'action_z_scale': 1.0,
        'gripper_state_mask_ratio': 0.0,

        'logbase': logbase,
        'prefix': 'diffusion/mimicgen_multitask_singleaction/',
        'exp_name': watch(args_to_watch),

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
        'n_reference': 1,
        'bucket': None,
        'device': 'cuda:0',
        'seed': 0,
        'renderer': 'utils.ParticleRenderer',
        'predict_epsilon': False,
        'env_config_dir': 'env_config/n_cubes',

        'loss_weights': None,
        'loss_discount': 1,

        'exe_steps': 3,
    },

    'plan': {
        'policy': 'sampling.GoalConditionedPolicy',
        'max_episode_length': 50,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:0',
        'seed': 0,
        'exe_steps': 3,

        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/mimicgen_multitask_singleaction/',
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

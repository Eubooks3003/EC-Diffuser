"""
Multitask multiview mimicgen @224 — SEMANTIC action + SEMANTIC proprio.

The token-grouping baseline: action splits pos/rot/grip = [3,3,1], proprio
splits pos/rot6d/grip = [3,6,1]. This is the tokenization the plain multitask
config used before commit 62180bf added `action_token_groups: 'per_dim'` and
turned that file into the uniform arm -- with both group keys unset,
`split_action_tokens` defaults to (gripper_dim > 0) = True, giving exactly these
splits.

Stated explicitly through the *grouped* path rather than by omitting the keys,
so all three arms (semantic / uniform / single-action) run the same generalized
code and differ only in grouping; the legacy named-module path would confound
tokenization with implementation.
"""

import os

from diffuser.utils import watch

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'seed'),
]

logbase = 'data'

# 12 d0 tasks, alphabetical → stable integer task IDs (0..11)
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

# Per-task rollout horizon. Sourced from each task's standalone config
# (diffuser/config/mimicgen_<task>_dlp.py 'mimicgen_max_steps' field). The
# multitask training config keeps one global default for backwards-compat
# (mode_to_args['12C_dlp']['mimicgen_max_steps']), but eval_paper.py reads
# this map via task_entries when --eval_task is set so each task gets its
# own horizon.
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


# The 224 store was rendered from the d1/d2 task variants, not the _d0 set the
# 84-res tokens used (only nut_assembly and pick_place stay d0). Tokens from the
# two sets are NOT interchangeable. Source: the store's own metadata.json.
TASK_VARIANT = {
    'coffee'                : 'coffee_d2',
    'coffee_preparation'    : 'coffee_preparation_d1',
    'hammer_cleanup'        : 'hammer_cleanup_d1',
    'kitchen'               : 'kitchen_d1',
    'mug_cleanup'           : 'mug_cleanup_d1',
    'nut_assembly'          : 'nut_assembly_d0',
    'pick_place'            : 'pick_place_d0',
    'square'                : 'square_d2',
    'stack'                 : 'stack_d1',
    'stack_three'           : 'stack_three_d1',
    'threading'             : 'threading_d2',
    'three_piece_assembly'  : 'three_piece_assembly_d2',
}


def _resolve_data_root():
    """Pick lambda (remote training) or local desktop (rollout) based on which exists."""
    candidates = [
        '/lambda/nfs/tal-lpwm-neurips-2026/data/mimicgen_224_wrist_tokens',
        '/home/ellina/Desktop/data/mimicgen_224_wrist_tokens',
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
    """One entry per task: pkl, calib h5, per-task DLP ckpt + cfg + max_steps."""
    entries = []
    for name in TASK_NAMES:
        task_dir = TASK_VARIANT[name]
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
        'multitask': True,                  # signals multitask path in dataset
        'task_entries': TASK_ENTRIES,
        'task_names': TASK_NAMES,
        'n_tasks': len(TASK_NAMES),
        'max_demos_per_task': 200,

        # Sentinel so setup.py's dataset-path resolver does not raise; the
        # multitask dataset ignores this and uses `task_entries` instead.
        'override_dataset_path': TASK_ENTRIES[0]['pkl'],
        # Renderer/eval default DLP if no --eval_task is given (used only for reference renders).
        'dlp_ckpt': TASK_ENTRIES[0]['dlp_ckpt'],
        'dlp_cfg':  TASK_ENTRIES[0]['dlp_cfg'],
        'dlp_ctor': 'models:DLP',
        'calib_h5_path': TASK_ENTRIES[0]['calib_h5'],

        # Shapes confirmed from per-task pkl meta (all 12 agree):
        # E=200, K=40 (20/view × 2 views), Dtok=10, A=7, G=10, BG=8 (4/view × 2 views)
        # Path lengths vary: pick_place_d0 max=798 (longest), stack_d0 min=81.
        'features_dim': 10,
        'gripper_dim': 10,
        'use_gripper_obs': True,
        'gripper_state_mask_ratio': 0.0,
        'bg_dim': 8,
        'use_bg_obs': True,
        'max_particles': 128,               # K=128 exactly (64/view x 2)
        'multiview': True,
        'device': 'cuda:0',

        # max_path_length must cover the longest task (pick_place_d0=798).
        # Set generously so per-episode `path_lengths` does the trimming.
        'max_path_length': 800,

        # In-training rollout eval is OFF: the GH200 training hosts have no sim
        # (robomimic/mujoco are not installed there), and eval_backend='mimicgen'
        # crashes at wiring time on ModuleNotFoundError before a single step.
        # Evaluate offline from checkpoints via
        # scripts/eval_paper_mimicgen_multitask.py. Matches the 0e6f26c
        # convention and keeps all three tokenization arms identical here.
        'eval_freq': 0,
        'eval_backend': 'none',
        'mimicgen_eval_episodes': 1,
        'n_steps_per_epoch': 500,

        # mimicgen rollout knobs (used only when --eval_task is set)
        # MUST match the views the tokens were encoded from: the 224 store is
        # agentview + robot0_eye_in_hand (wrist), NOT the agentview + sideview
        # the 84-res d0 tokens used. Rendering sideview here would feed wrist
        # token slots with sideview particles -- no crash, just a silent
        # observation-space mismatch that reads as policy failure.
        # Provenance: meta['cameras'] in mimicgen_224_wrist_tokens/*/*.pkl.
        # Render at 224 so it matches the training memmap natively rather than
        # going through a 256->224 resize.
        # The 224 store was rendered natively at 224; eval must render at the
        # same size or the DLP sees upsampled frames it never trained on.
        'mimicgen_preprocess_render_res': 224,
        'mimicgen_cams': ['agentview', 'robot0_eye_in_hand'],
        'mimicgen_camera_width': 224,
        'mimicgen_camera_height': 224,
        'mimicgen_max_steps': 600,
        'mimicgen_pixel_stride': 1,
        'use_absolute_actions': False,

        # diffusion knobs
        'horizon': 16,
        'exe_steps': 8,
        'random_init': True,
        'random_init_eval': True,
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

        'max_particles': 128,
        'positional_bias': False,
        'multiview': True,

        # Per-dimension action/proprio tokenization: one token per scalar dim.
        # 'per_dim' expands to [1]*action_dim and [1]*gripper_dim at build time,
        # so the same value stays uniform with the RLBench policy.
        'action_token_groups': [3, 3, 1],
        'proprio_token_groups': [3, 6, 1],

        # multitask flags (defaults; overridden by mode_to_args)
        'multitask': False,
        'n_tasks': 1,

        # dataset
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

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/mimicgen224_multitask_semantic/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2.5e6,    # 5000 epochs @ 500 steps/epoch (was 2e6 = 4000); +1000 epochs to continue training
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
        'prefix': 'plans/mimicgen224_multitask_semantic/',
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

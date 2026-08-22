"""
Multitask DexJoCo MERGED single+bimanual @84 — hand-as-ONE-token action + proprio.

Layouts below were confirmed empirically from the store's `_lowdim.npz`, not
assumed: quaternion slices were located by unit-norm test and action blocks by
correlation against the matching proprio block. NOTE the action and proprio
orderings differ (poses are grouped in proprio but per-arm interleaved in action) -- getting them backwards mis-tokenizes silently
without crashing.

    action  (44-D, ABSOLUTE rotvec) = [r_xyz(3), r_rotvec(3), r_allegro(16), l_xyz(3), l_rotvec(3), l_allegro(16)]
    proprio (46-D)                  = [r_xyz(3), r_quat(4), l_xyz(3), l_quat(4), r_allegro(16), l_allegro(16)]

The whole 16-dim Allegro hand is ONE token here, rather than one token per
digit. This is the direct ablation counterpart to the *_semantic_* configs
(which use [.., 4,4,4,4]): identical in every other respect, so the pair
isolates what per-finger tokenization buys. Proprio moves in step so the hand
grouping is the only delta.

All 11 tasks in ONE policy. Single-arm is zero-padded up into the bimanual
interface, which is DexJoCo's own convention -- see `pad_state_dim46` in
dp_dexjoco_env.py ("Pad to 46 dims for the shared model interface") and their
44-D action head (convert_to_action_dim_44_model.py). The padding is
semantically clean rather than a hack: the bimanual layout leads with the RIGHT
arm, so single-arm == right arm == the first 22 action / 23 proprio dims, and a
single-arm rollout simply executes action[:22].

    single  -> bimanual     action 22->44   proprio 23->46   bg 8->12   K 40->60

ReplayBuffer.load_paths_from_pickles zero-pads every trailing axis to the widest
task, and DatasetNormalizer fits the padded dimensions on full-width rows only
(fit_normalizer_masked) so single-arm zeros do not drag the bimanual statistics
toward zero. One DLP checkpoint encodes both stores, so the particle latent
space is already common across all 11 tasks.

In-training rollout eval is OFF and there is no DexJoCo eval backend at all:
`eval_backend='mimicgen'` would not apply here. Evaluate offline from
checkpoints.
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

# Alphabetical → stable integer task IDs (0..10)
TASK_NAMES = [
    'bimanual_assembly',
    'bimanual_hanoi',
    'bimanual_microwave_cook',
    'bimanual_photograph',
    'bimanual_unlock_ipad',
    'click_mouse',
    'fold_glasses',
    'hammer_nail',
    'pick_bucket',
    'pinch_tongs',
    'water_plant',
]
TASK_NAME_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}


def _resolve_data_root():
    """Pick lambda (remote training) or local desktop, whichever exists."""
    candidates = [
        '/lambda/nfs/tal-lpwm-neurips-2026/data/dexjoco_84_tokens',
        '/home/ellina/Desktop/data/dexjoco_84_tokens',
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


DATA_ROOT = _resolve_data_root()


def _build_task_entries():
    """One entry per task: pkl + the DLP ckpt/cfg snapshotted beside it."""
    entries = []
    for name in TASK_NAMES:
        entries.append({
            'name': name,
            'task_id': TASK_NAME_TO_ID[name],
            'pkl': os.path.join(DATA_ROOT, name, f'{name}.pkl'),
            'dlp_ckpt': os.path.join(DATA_ROOT, name, 'dlp_ckpt.pt'),
            'dlp_cfg':  os.path.join(DATA_ROOT, name, 'dlp_config.json'),
        })
    return entries


TASK_ENTRIES = _build_task_entries()


# Key must match the mode setup.py computes: "{num_entity}C_{input_type}".
# Select with --num_entity 11.
mode_to_args = {
    '11C_dlp': {
        'dataset': 'multitask',
        'multitask': True,
        'task_entries': TASK_ENTRIES,
        'task_names': TASK_NAMES,
        'n_tasks': len(TASK_NAMES),
        'max_demos_per_task': 100,          # every task ships exactly 100 episodes

        # Sentinel so setup.py's path resolver does not raise; the multitask
        # dataset ignores it and uses `task_entries`.
        'override_dataset_path': TASK_ENTRIES[0]['pkl'],
        'dlp_ckpt': TASK_ENTRIES[0]['dlp_ckpt'],
        'dlp_cfg':  TASK_ENTRIES[0]['dlp_cfg'],
        'dlp_ctor': 'models:DLP',

        # Widest-task shapes; narrower single-arm tasks are zero-padded to these.
        # bimanual: K=60 (20/view x 3), A=44, G=46, BG=12
        # single:   K=40 (20/view x 2), A=22, G=23, BG=8   -> padded
        'features_dim': 10,
        'gripper_dim': 46,
        'use_gripper_obs': True,
        'gripper_state_mask_ratio': 0.0,
        'bg_dim': 12,
        'use_bg_obs': True,
        'max_particles': 64,              # covers K=60
        'multiview': True,
        'device': 'cuda:0',

        # Longest episode across all 11 tasks is 1422 (bimanual_hanoi); per-episode
        # `path_lengths` does the real trimming.
        'max_path_length': 1450,

        # No sim on the training host and no DexJoCo eval backend exists.
        'eval_freq': 0,
        'eval_backend': 'none',
        'n_steps_per_epoch': 500,

        # DexJoCo actions are ABSOLUTE world-frame targets (mocap pos/quat +
        # direct Allegro joint targets), unlike mimicgen's relative deltas.
        'use_absolute_actions': True,

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

        'max_particles': 64,
        'positional_bias': False,
        'multiview': True,

        # Token grouping (see pint.py grouped_tokens): each group is one
        # transformer token with its own projection/type-encoding/decoder.
        # [n]=one token spanning n dims; 'per_dim'=one token per scalar.
        'action_token_groups': [3, 3, 16, 3, 3, 16],
        'proprio_token_groups': [3, 4, 3, 4, 16, 16],

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
        'max_path_length': 1450,
        'obs_only': False,
        'action_only': False,
        'action_z_scale': 1.0,
        'gripper_state_mask_ratio': 0.0,

        # serialization
        'logbase': logbase,
        'prefix': 'diffusion/dexjoco84_merged_multitask_handtoken/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 200,
        'loss_type': 'l1',
        'n_train_steps': 2.5e6,
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
        'prefix': 'plans/dexjoco84_merged_multitask_handtoken/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 10,
        'max_render': 8,
    },
}

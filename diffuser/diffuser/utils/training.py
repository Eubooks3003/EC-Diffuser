import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import wandb
from tqdm import tqdm
try:
    from dlp_utils import log_rgb_voxels
except ImportError:
    log_rgb_voxels = None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, front_bg=None, side_bg=None, latent_rep_model=None):
        timer = Timer()
        for step in range(int(n_train_steps)):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)

                log_dict = {'step': self.step, 'loss': loss, **infos}

                # lightweight offline eval every log
                # eval_stats = self.eval_offline_goal_metrics(
                #     n_batches=5,     # keep small so it’s cheap
                #     goal_tau=2.0
                # )
                # print("eval_stats: ", eval_stats  )
                # log_dict.update(eval_stats)

                wandb.log(log_dict)


            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference, front_bg=front_bg, side_bg=side_bg)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     self.render_samples(front_bg=front_bg, side_bg=side_bg)

            self.step += 1

    def evaluate(self, n_eval_steps):
        '''
            evaluate model on validation set
        '''
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            all_losses = []
            all_infos = []
            for step in tqdm(range(n_eval_steps), desc='eval'):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss, infos = self.model.loss(*batch)
                all_losses.append(loss.item())
                all_infos.append(infos)
            for key in infos:
                mean_val = np.mean([info[key].cpu().numpy() for info in all_infos])
                print(f'{key}: {mean_val:8.4f}')
                wandb.log({'step':self.step, key: mean_val})
            print(f'loss: {np.mean(all_losses):8.4f}')
            wandb.log({'step':self.step, 'loss': np.mean(all_losses)})

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        ckpt_dir = os.path.join(self.logdir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        savepath = os.path.join(ckpt_dir, f'state_{epoch}_step{self.step}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        import glob
        ckpt_dir = os.path.join(self.logdir, 'ckpt')

        # Try new location + naming first (ckpt/state_{epoch}_step{step}.pt)
        pattern = os.path.join(ckpt_dir, f'state_{epoch}_step*.pt')
        matches = glob.glob(pattern)
        if matches:
            loadpath = sorted(matches)[-1]
        else:
            # Try new naming in old location (state_{epoch}_step{step}.pt)
            pattern = os.path.join(self.logdir, f'state_{epoch}_step*.pt')
            matches = glob.glob(pattern)
            if matches:
                loadpath = sorted(matches)[-1]
            else:
                # Fall back to old naming convention (state_{epoch}.pt)
                loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')

        data = torch.load(loadpath)
        print(f'[ utils/training ] Loaded model from {loadpath}', flush=True)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])

    def load_latest(self):
        '''
            finds the latest checkpoint in logdir/ckpt and loads it.
            returns True if a checkpoint was loaded, False otherwise.
        '''
        import glob as glob_mod
        ckpt_dir = os.path.join(self.logdir, 'ckpt')
        pattern = os.path.join(ckpt_dir, 'state_*_step*.pt')
        matches = glob_mod.glob(pattern)
        if not matches:
            return False

        def _get_step(path):
            return int(os.path.basename(path).split('_step')[1].replace('.pt', ''))

        latest = max(matches, key=_get_step)
        data = torch.load(latest, map_location='cpu')
        print(f'[ utils/training ] Resuming from {latest}', flush=True)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
        return True

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10, front_bg=None, side_bg=None):
        '''
            renders training points
        '''
        if batch_size <= 0:
            return  # reference rendering disabled

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        # Trajectory format: [actions, gripper_state, bg_features, observations]
        gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
        bg_dim = getattr(self.dataset, 'bg_dim', 0)
        action_dim = self.dataset.action_dim

        # Extract bg_features if present
        bg_features_seq = None
        if bg_dim > 0:
            bg_start_idx = action_dim + gripper_dim
            bg_end_idx = bg_start_idx + bg_dim
            normed_bg = trajectories[:, :, bg_start_idx:bg_end_idx]
            bg_features_seq = self.dataset.normalizer.unnormalize(normed_bg, 'bg_features')

        obs_start_idx = action_dim + gripper_dim + bg_dim
        normed_observations = trajectories[:, :, obs_start_idx:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(
            savepath, observations,
            front_bg=front_bg, side_bg=side_bg,
            bg_features_seq=bg_features_seq,
            log_bg=(bg_dim > 0), log_full=(bg_dim > 0)
        )

    def render_samples(self, batch_size=2, n_samples=2, front_bg=None, side_bg=None):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + gripper_dim + bg_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## Extract dimensions
            gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
            bg_dim = getattr(self.dataset, 'bg_dim', 0)
            action_dim = self.dataset.action_dim

            ## Extract bg_features if present
            bg_features_seq = None
            if bg_dim > 0:
                bg_start_idx = action_dim + gripper_dim
                bg_end_idx = bg_start_idx + bg_dim
                normed_bg = trajectories[:, :, bg_start_idx:bg_end_idx]
                # Get condition bg_features for prepending
                # Conditions include [gripper_state, bg_features, observations]
                cond_gripper_dim = gripper_dim
                cond_bg_start = cond_gripper_dim
                cond_bg_end = cond_bg_start + bg_dim
                normed_cond_bg = to_np(batch.conditions[0])[:, cond_bg_start:cond_bg_end][:, None]
                # Prepend condition bg to trajectory bg
                normed_bg = np.concatenate([
                    np.repeat(normed_cond_bg, n_samples, axis=0),
                    normed_bg
                ], axis=1)
                bg_features_seq = self.dataset.normalizer.unnormalize(normed_bg, 'bg_features')

            ## [ n_samples x horizon x observation_dim ]
            obs_start_idx = action_dim + gripper_dim + bg_dim
            normed_observations = trajectories[:, :, obs_start_idx:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]
            # Extract just observations from conditions (skip gripper and bg)
            cond_obs_start = gripper_dim + bg_dim
            normed_cond_obs = normed_conditions[:, :, cond_obs_start:]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_cond_obs, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(
                savepath, observations,
                front_bg=front_bg, side_bg=side_bg,
                bg_features_seq=bg_features_seq,
                log_bg=(bg_dim > 0), log_full=(bg_dim > 0)
            )
    

    @torch.no_grad()
    def eval_mimicgen_rollouts(
        self,
        make_env_fn,
        dlp_model,
        calib_h5_path,
        n_episodes=5,
        max_steps=500,
        bounds_xyz=((-2,2), (-2,2), (-0.2,2.5)),
        grid_dhw=(64,64,64),
        cams=("agentview","sideview","robot0_eye_in_hand"),
        pixel_stride=2,
        goal_from_env_fn=None,
        goal_provider=None,         # NEW: DatasetGoalProvider for init_state + goal pairing
        random_init=False,          # If True, use random env reset instead of dataset init states
        task=None,                  # Task name for task-specific voxel bounds (e.g., "threading", "hammer_cleanup")

        save_videos=True,
        video_dir=None,
        video_fps=20,
        video_cams=("agentview",),

        renderer_3d=None,
        render_debug=True,
        render_debug_steps=(0,),
        exe_steps=1,  # NEW: number of actions to execute from each plan before replanning
        log_imagined_states=True,  # NEW: decode and log diffuser's predicted future states
        log_imagined_episode=0,  # which episode to log imagined states for
        log_imagined_plan_idx=0,  # which plan (replan) within the episode to log
    ):
        """
        True success eval by stepping MimicGen.
        - make_env_fn: () -> env
        - dlp_model: your 3D DLP (already loaded)
        - calib_h5_path: hdf5 with meta/cameras/*
        - goal_from_env_fn: (env, raw_obs) -> raw_goal_obs (optional, legacy)
        - goal_provider: DatasetGoalProvider for paired init_state + goal tokens (recommended)
        - exe_steps: number of actions to execute from each predicted trajectory before replanning
                     (action chunking). Default=1 means replan every step.
        - log_imagined_states: if True, decode and log the diffuser's predicted future observations
                               to wandb using the 3D renderer. This allows verifying that the
                               imagined states reconstruct into something reasonable.
        - log_imagined_episode: which episode to log imagined states for (default 0 = first episode)
        - log_imagined_plan_idx: which plan within the episode to log (default 0 = first plan)
        """
        from diffuser.envs.mimicgen_dlp_wrapper import MimicGenDLPWrapper

        device = next(self.ema_model.parameters()).device
        dlp_model = dlp_model.to(device).eval()

        import os
        import numpy as np
        import imageio.v2 as imageio

        def _to_uint8(img):
            img = np.asarray(img)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            # CHW -> HWC if needed
            if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[-1] not in (1,3,4):
                img = np.transpose(img, (1,2,0))
            if img.dtype != np.uint8:
                if img.max() <= 1.5:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        def _frame_from_raw_obs(raw_obs, cams_to_use):
            frames = []
            for cam in cams_to_use:
                k = f"{cam}_image"
                if k not in raw_obs:
                    continue
                frames.append(_to_uint8(raw_obs[k]))
            if not frames:
                return None
            # concat multi-view horizontally
            if len(frames) == 1:
                return frames[0]
            # make same height if needed (simple crop)
            h = min(f.shape[0] for f in frames)
            frames = [f[:h] for f in frames]
            return np.concatenate(frames, axis=1)
        def _render_tokens_debug(tag, obs_vec_flat, horizon_step=None):
            if renderer_3d is None:
                return
            if obs_vec_flat is None:
                raise RuntimeError("obs_vec_flat is None")

            # obs_vec_flat is 1D = K*Dtok
            K = getattr(envw, "K", None)
            Dtok = getattr(envw, "Dtok", None)

            print("K: ", K)
            print("Dtok: ", Dtok)


            # TODO: Actually make this read from something
            if K is None:
                K = 16
            if Dtok is None:
                # hard fail-fast: you should set this; but infer if possible
                if obs_vec_flat.size % K != 0:
                    raise RuntimeError(f"Cannot infer Dtok: len(obs_vec_flat)={obs_vec_flat.size} not divisible by K={K}")
                Dtok = obs_vec_flat.size // K
            
            print("obs vec flat: ", obs_vec_flat.shape)
            toks = np.asarray(obs_vec_flat, dtype=np.float32).reshape(1, K, Dtok)  # [1,K,Dtok]

            # Use monotonic step for wandb: prefer global trainer step if provided.
            step_to_log = 200
            print("toks: ", toks.shape)
            renderer_3d.render(
                toks[0],                              # renderer accepts [K,Dtok] or flat; yours accepts either
                tag=tag,
                step=step_to_log,
                base="eval_debug"
            )


        if save_videos:
            if video_dir is None:
                video_dir = os.path.join(self.logdir, "eval_videos", f"step_{self.step}")
            os.makedirs(video_dir, exist_ok=True)


        successes = []
        returns = []
        lengths = []

        # Print eval configuration
        print("=" * 60)
        print(f"[eval_mimicgen_rollouts] Starting {n_episodes} episodes, max_steps={max_steps}")
        print(f"[eval_mimicgen_rollouts] Goal provider: {'DatasetGoalProvider' if goal_provider is not None else 'None (legacy mode)'}")
        print(f"[eval_mimicgen_rollouts] Init mode: {'RANDOM' if random_init else 'DATASET (fixed init states)'}")
        print(f"[eval_mimicgen_rollouts] ACTION CHUNKING: exe_steps={exe_steps}, horizon={self.dataset.horizon}")
        if exe_steps > 1:
            print(f"[eval_mimicgen_rollouts] Will execute {exe_steps} actions per plan before replanning")
        else:
            print(f"[eval_mimicgen_rollouts] WARNING: exe_steps=1 means replanning every step (no chunking)")
        if log_imagined_states and renderer_3d is not None:
            print(f"[eval_mimicgen_rollouts] IMAGINED STATE LOGGING: enabled for ep={log_imagined_episode}, plan={log_imagined_plan_idx}")
        elif log_imagined_states and renderer_3d is None:
            print(f"[eval_mimicgen_rollouts] IMAGINED STATE LOGGING: requested but renderer_3d is None, skipping")
        print("=" * 60, flush=True)

        # Create env and wrapper ONCE to avoid OpenGL context corruption
        env = make_env_fn()
        envw = MimicGenDLPWrapper(
            env=env,
            dlp_model=dlp_model,
            device=device,
            cams=cams,
            grid_dhw=grid_dhw,
            pixel_stride=pixel_stride,
            calib_h5_path=calib_h5_path,
            get_goal_raw_obs_fn=goal_from_env_fn,
            goal_provider=goal_provider,  # NEW: dataset-based goal provider
            random_init=random_init,      # NEW: random vs dataset init
            normalize_to_unit_cube=False,
            task=task,                    # Task name for task-specific voxel bounds
        )

        for ep in range(n_episodes):
            print(f"\n[eval] Episode {ep+1}/{n_episodes}")

            obs_vec = envw.reset()
            # envw.print_params_like_h5_script(envw.last_raw_obs)   

            ep_ret = 0.0
            plan_idx = 0  # track how many times we've replanned this episode
            frames = []
            if save_videos and envw.last_raw_obs is not None:
                fr = _frame_from_raw_obs(envw.last_raw_obs, video_cams)
                if fr is not None:
                    frames.append(fr)
            # ACTION CHUNKING: track planned actions buffer
            action_buffer = None  # will hold (H, a_dim) tensor of planned actions
            action_idx = 0  # which action in buffer to execute next
            a_dim = self.dataset.action_dim
            z_scale = getattr(self.dataset, 'action_z_scale', 1.0)

            # Helper to normalize observations
            def norm_obs(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "observations")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Helper to normalize gripper state
            def norm_gripper(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "gripper_state")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Helper to normalize bg_features
            def norm_bg(v):
                v_norm = self.dataset.normalizer.normalize(v[None], "bg_features")[0]
                return torch.from_numpy(v_norm).float().to(device)

            # Check if we should use gripper observations and bg_features
            use_gripper_obs = getattr(self.dataset, 'use_gripper_obs', False)
            gripper_dim = getattr(self.dataset, 'gripper_dim', 0)
            use_bg_obs = getattr(self.dataset, 'use_bg_obs', False)
            bg_dim = getattr(self.dataset, 'bg_dim', 0)

            t = 0
            while t < max_steps:
                # Check if we need to replan (no buffer, or exhausted exe_steps actions)
                need_replan = (action_buffer is None) or (action_idx >= exe_steps) or (action_idx >= action_buffer.shape[0])

                if need_replan:
                    # ====== PLANNING PHASE ======
                    # Build current observation condition
                    # Condition format: [gripper_state (optional), bg_features (optional), observations]
                    cond_parts = []

                    # Add gripper state if enabled
                    if use_gripper_obs and gripper_dim > 0:
                        gripper_cond = envw.get_gripper_cond(horizon=self.dataset.horizon)
                        if gripper_cond is not None and 0 in gripper_cond:
                            gripper_norm = norm_gripper(gripper_cond[0])
                            cond_parts.append(gripper_norm)

                    # Add bg_features if enabled
                    if use_bg_obs and bg_dim > 0:
                        bg_cond = envw.get_bg_cond(horizon=self.dataset.horizon)
                        if bg_cond is not None and 0 in bg_cond:
                            bg_norm = norm_bg(bg_cond[0])
                            cond_parts.append(bg_norm)

                    # Add observations
                    obs_norm = norm_obs(obs_vec)
                    cond_parts.append(obs_norm)

                    # Concatenate all parts
                    cond_0 = torch.cat(cond_parts, dim=-1)[None, :]

                    # No goal conditioning — only condition on initial observation.
                    # GoalDataset trains with conditions={0: obs_0} only (no H-1 key),
                    # so inference must match: adding a zero condition at H-1 would
                    # force the last observation to the mean state at every denoising
                    # step, distorting the predicted trajectory.
                    cond = {
                        0: cond_0,
                    }

                    if t == 0:
                        print(f"\n[EVAL] ep={ep}: No goal conditioning (matches training)")
                        print(f"  cond_0 shape: {cond_0.shape}")

                    # Sample new trajectory from diffusion model
                    sample = self.ema_model(cond, verbose=False)
                    traj = sample.trajectories[0]  # (H, action_dim + obs_dim)
                    action_buffer = traj[:, :a_dim].detach().cpu().numpy().astype(np.float32)  # (H, a_dim)
                    action_idx = 0

                    # Print planning info
                    n_actions_to_exec = min(exe_steps, action_buffer.shape[0])
                    print(f"[PLAN] t={t}: Generated {action_buffer.shape[0]}-step plan, will execute {n_actions_to_exec} actions")
                    if t < 10:  # Show full planned trajectory for first few plans
                        print(f"  Planned Z trajectory (normalized): {[f'{action_buffer[i, 2]:.3f}' for i in range(min(5, action_buffer.shape[0]))]}")

                    # (3D imagined state logging removed — not applicable for 2D DLP)

                    plan_idx += 1  # increment plan counter

                # ====== EXECUTION PHASE ======
                # Get current action from buffer
                a_norm = action_buffer[action_idx]

                # Unnormalize action
                a = self.dataset.normalizer.unnormalize(a_norm[None], "actions")[0]

                # Unscale Z if action_z_scale was applied during training
                if z_scale != 1.0:
                    a[2] /= z_scale

                # Step environment
                obs_vec, r, done, info = envw.step(a)

                # Print execution info
                if t < 10:
                    print(f"[EXEC] t={t}: action_idx={action_idx}/{exe_steps} | pos=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}] grip={a[6]:.2f}")
                    if t == 0:
                        norm = self.dataset.normalizer.normalizers.get('actions', None)
                        if norm is not None:
                            if hasattr(norm, 'mins') and hasattr(norm, 'maxs'):
                                print(f"  normalizer mins: {norm.mins.flatten()}")
                                print(f"  normalizer maxs: {norm.maxs.flatten()}")
                            elif hasattr(norm, 'means') and hasattr(norm, 'stds'):
                                print(f"  normalizer means: {norm.means.flatten()}")
                                print(f"  normalizer stds: {norm.stds.flatten()}")

                action_idx += 1
                ep_ret += float(r)

                if save_videos and envw.last_raw_obs is not None:
                    fr = _frame_from_raw_obs(envw.last_raw_obs, video_cams)
                    if fr is not None:
                        frames.append(fr)

                t += 1
                # Check for episode termination (done) or success
                episode_success = info.get("success", False) if isinstance(info, dict) else False
                if done or episode_success:
                    if episode_success:
                        print(f"[EVAL] Episode succeeded at t={t}, stopping early")
                    break

            # ---- new: write video ----
            if save_videos and len(frames) > 0:
                out_path = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
                # macro_block_size=None avoids ffmpeg issues with non-multiple-of-16 sizes
                imageio.mimsave(out_path, frames, fps=int(video_fps), macro_block_size=None)
                print(f"[mimicgen eval] wrote video: {out_path}", flush=True)
            # success flag (wrapper tries common info keys; may be None)
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            successes.append(success)
            returns.append(ep_ret)
            lengths.append(t + 1)

        try:
            env.close()
        except Exception:
            pass

        out = {
            "sim/success_rate": float(np.mean(successes)) if len(successes) else 0.0,
            "sim/avg_return": float(np.mean(returns)) if len(returns) else 0.0,
            "sim/avg_len": float(np.mean(lengths)) if len(lengths) else 0.0,
        }
        return out

    @torch.no_grad()
    def eval_rlbench_rollouts(
        self,
        make_env_fn,
        make_policy_fn,
        n_episodes: int = 5,
        max_steps: int = 400,
        task_name: str = None,
        exe_steps: int = 1,
    ):
        """
        Live RLBench rollouts with the trained language-conditioned 2D-DLP policy.

        Mirrors the per-component-normalize-then-concat pattern used by the
        working mimicgen eval, rather than routing through the Policy wrapper
        (which doesn't know about bg_features in its _format_conditions path).

        - make_env_fn:    () -> RLBenchDLPEnv
        - make_policy_fn: () -> LanguageConditionedPolicy  (used ONLY for its
                          set_instruction() + CLIP encoder; the actual sampling
                          is done via self.ema_model directly below).
        - exe_steps:      Number of predicted actions a[1..exe_steps] to execute
                          per plan before re-encoding and re-sampling. a[0] is
                          skipped because it is pinned to the current gripper
                          pose by apply_conditioning. Override at runtime with
                          ECDIFF_EXE_STEPS=<int>.
        """
        import torch as _torch
        device = next(self.ema_model.parameters()).device

        _env_exe_steps = os.environ.get("ECDIFF_EXE_STEPS")
        if _env_exe_steps is not None:
            try:
                exe_steps = int(_env_exe_steps)
            except ValueError:
                print(f"[eval_rlbench] invalid ECDIFF_EXE_STEPS={_env_exe_steps!r}; ignoring", flush=True)
        exe_steps = max(1, int(exe_steps))

        env = make_env_fn()
        policy = make_policy_fn()

        a_dim = int(self.ema_model.action_dim)
        gripper_dim = int(getattr(self.ema_model.model, "gripper_dim", 0))
        bg_dim = int(getattr(self.ema_model.model, "bg_dim", 0))
        K = int(getattr(self.ema_model.model, "max_particles", 0))
        D = int(getattr(self.ema_model.model, "features_dim", 0))

        def norm_obs(tok):
            # tok: (K, D) np -> (1, K*D) torch
            normed = self.dataset.normalizer.normalize(tok, "observations")
            return _torch.from_numpy(normed).float().reshape(1, -1)

        def norm_bg(bg_vec):
            if bg_dim == 0:
                return None
            normed = self.dataset.normalizer.normalize(bg_vec.reshape(1, -1), "bg_features")
            return _torch.from_numpy(normed).float()

        def norm_gripper(gs_vec):
            if gripper_dim == 0:
                return None
            normed = self.dataset.normalizer.normalize(gs_vec.reshape(1, -1), "gripper_state")
            return _torch.from_numpy(normed).float()

        successes = []
        lengths = []
        per_variation = {}

        # OCGS-style GT replay: save expert-demo video alongside policy rollout,
        # both starting from the same initial scene via task.reset_to_demo(demo).
        # Enabled when ECDIFF_SAVE_GT_VIDEO=1 and ECDIFF_DEMO_ROOT is set to the
        # raw RLBench dataset root (with <task>/all_variations/episodes/episodeN/).
        _save_gt = os.environ.get("ECDIFF_SAVE_GT_VIDEO") == "1"
        _demo_root = os.environ.get("ECDIFF_DEMO_ROOT")
        _pkl_actions_all = None
        _pkl_vars = None
        if _save_gt:
            if not _demo_root:
                print("[eval_rlbench] WARN: ECDIFF_SAVE_GT_VIDEO=1 but ECDIFF_DEMO_ROOT unset; GT replay disabled", flush=True)
                _save_gt = False
            else:
                _pkl_path = (
                    getattr(self.dataset, "dataset_path", None)
                    or os.environ.get("ECDIFF_PKL_PATH")
                )
                if _pkl_path and os.path.isfile(_pkl_path):
                    import pickle as _pickle
                    with open(_pkl_path, "rb") as _f:
                        _pkl = _pickle.load(_f)
                    _pkl_actions_all = np.asarray(_pkl["actions"])
                    _pkl_vars = np.asarray(_pkl.get("variation_number",
                                                    np.zeros(_pkl_actions_all.shape[0], dtype=int)))
                    print(f"[eval_rlbench] GT replay: {_pkl_actions_all.shape[0]} demos from {_pkl_path}", flush=True)
                else:
                    print(f"[eval_rlbench] WARN: pkl not found ({_pkl_path!r}); GT replay disabled", flush=True)
                    _save_gt = False

        def _load_demo_from_disk(ep_idx):
            """Load a raw RLBench Demo from low_dim_obs.pkl for reset_to_demo."""
            import pickle as _pickle
            ep_dir = os.path.join(
                _demo_root, task_name, "all_variations", "episodes", f"episode{ep_idx}"
            )
            with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as _f:
                demo = _pickle.load(_f)
            _var_path = os.path.join(ep_dir, "variation_number.pkl")
            if os.path.isfile(_var_path):
                with open(_var_path, "rb") as _f:
                    demo.variation_number = int(_pickle.load(_f))
            return demo

        # Imagined-state visualization: dump the model's predicted particle tokens
        # for the step just executed, overlay on the real post-step RGB, save as
        # its own video. Lets us tell apart "action predictions correct / dynamics
        # imagined correctly" from "actions right but no scene understanding".
        _save_imagined = os.environ.get("ECDIFF_SAVE_IMAGINED") == "1"
        _save_imagined_recon = os.environ.get("ECDIFF_SAVE_IMAGINED_RECON") == "1"
        _obs_start = a_dim + gripper_dim + bg_dim  # particle tokens begin here in traj
        _bg_start = a_dim + gripper_dim            # bg features begin here
        _V = max(len(getattr(env, "cams", []) or []), 1)
        _K_per_view = max(K // _V, 1)
        _bg_per_view = max(bg_dim // _V, 1) if bg_dim > 0 else 0
        _dlp_model = getattr(env, "_dlp_model", None)
        if _save_imagined_recon and _dlp_model is None:
            print("[eval_rlbench] WARN: ECDIFF_SAVE_IMAGINED_RECON=1 but env._dlp_model not set; disabling", flush=True)
            _save_imagined_recon = False

        print("=" * 60)
        print(f"[eval_rlbench] task={task_name} n_episodes={n_episodes} max_steps={max_steps}")
        print(f"[eval_rlbench] a_dim={a_dim} gripper_dim={gripper_dim} bg_dim={bg_dim} K={K} D={D}")
        print(f"[eval_rlbench] action chunking: exe_steps={exe_steps}")
        print(f"[eval_rlbench] save_gt={_save_gt} save_imagined={_save_imagined}")
        print("=" * 60, flush=True)

        for ep in range(n_episodes):
            # GT replay: pick a demo to anchor this episode's initial state
            _demo = None
            _demo_idx = None
            if _save_gt and _pkl_actions_all is not None:
                _demo_idx = ep % _pkl_actions_all.shape[0]
                try:
                    _demo = _load_demo_from_disk(_demo_idx)
                except Exception as e:
                    print(f"[eval_rlbench] ep={ep}: demo {_demo_idx} load failed "
                          f"({type(e).__name__}: {str(e)[:160]}); falling back to random reset", flush=True)
                    _demo = None

            try:
                obs_dict = env.reset(demo=_demo) if _demo is not None else env.reset()
            except Exception as e:
                print(f"[eval_rlbench] ep={ep} reset failed: {type(e).__name__}: {e}", flush=True)
                successes.append(0.0); lengths.append(0); continue

            _lang_override = os.environ.get("ECDIFF_LANG_OVERRIDE")
            if _lang_override:
                print(f"[eval_rlbench]   LANG OVERRIDE: replacing {obs_dict['language']!r} with {_lang_override!r}", flush=True)
                policy.set_instruction(_lang_override)
            else:
                policy.set_instruction(obs_dict["language"])
            # Broadcast lang tokens + mask to batch=1, move to device.
            lang = policy._lang_tokens.to(device)[:1]
            lang_mask = policy._lang_mask.to(device)[:1]

            last_reward = 0.0
            last_error = None
            done = False
            t = 0
            _printed_token_stats = False
            # ACTION CHUNKING: mirror the OCGS receding-horizon pattern.
            # Each plan produces an H-step trajectory; a[0] is pinned to the
            # current pose, so we buffer a[1..H-1] and play out up to
            # exe_steps of them before re-encoding and re-sampling.
            action_buffer = None  # (H-1, a_dim) normalized actions from last plan
            chunk_idx = 0
            last_traj = None      # retained for debug-action printing
            last_gs_np = None
            # Per-step imagined particle XY (front view only) for the step just
            # executed. Populated when ECDIFF_SAVE_IMAGINED=1; one entry per step
            # aligned with env frame indices [1:], since frames[0] is pre-step.
            _imagined_kps_front = []
            # Per-step DLP decoder RGB reconstruction of imagined particle state.
            _imagined_recon_frames = []
            _imagined_recon_error_printed = False
            # Per-step DLP decoder RGB reconstruction of the LIVE (ground-truth)
            # particle state. Isolates decoder-pipeline correctness from model
            # prediction quality: if this reconstructs the real scene but the
            # imagined one doesn't, the model's obs predictions are the problem.
            _live_recon_frames = []
            _live_recon_error_printed = False
            while not done and t < max_steps:
                need_replan = (
                    action_buffer is None
                    or chunk_idx >= exe_steps
                    or chunk_idx >= action_buffer.shape[0]
                )

                if need_replan:
                    tokens = np.asarray(obs_dict["obs"], dtype=np.float32)
                    if tokens.ndim == 1:
                        tokens = tokens.reshape(K, D)
                    if os.environ.get("ECDIFF_ABLATE_TOKENS") == "1":
                        # Replace tokens with Gaussian noise matching training-data stats
                        # (~N(0, 1.5^2) based on inspection). If rollout behavior is
                        # ~indistinguishable from the real run, the policy is not using
                        # the particle tokens.
                        tokens = np.random.randn(*tokens.shape).astype(np.float32) * 1.5
                    if not _printed_token_stats:
                        print(f"[eval_rlbench] live token stats ep={ep} t={t}: "
                              f"z=[{tokens[:,0].min():+.3f},{tokens[:,0].max():+.3f}] "
                              f"depth=[{tokens[:,4].min():+.3f},{tokens[:,4].max():+.3f}] "
                              f"feat_mean={tokens[:,6:].mean():+.3f} feat_std={tokens[:,6:].std():.3f}",
                              flush=True)
                        _printed_token_stats = True
                    bg_np = np.asarray(obs_dict["bg_features"], dtype=np.float32)
                    gs_np = np.asarray(obs_dict["gripper_state"], dtype=np.float32)

                    parts = []
                    if gripper_dim > 0:
                        parts.append(norm_gripper(gs_np))
                    if bg_dim > 0:
                        parts.append(norm_bg(bg_np))
                    parts.append(norm_obs(tokens))
                    cond_0 = _torch.cat(parts, dim=-1).to(device)   # (1, gripper+bg+K*D)
                    cond = {0: cond_0}

                    # Proprioception conditioning: pin action[0] = current gripper pose
                    # (same 10D [pos, rot6d, open] format as actions) in actions-normalized space.
                    action_cond_raw = gs_np.reshape(1, -1)  # (1, 10) — gripper_state matches action dim
                    action_cond_normed = self.dataset.normalizer.normalize(action_cond_raw, "actions")
                    action_cond_tensor = _torch.from_numpy(action_cond_normed).float().to(device)
                    action_cond = {0: action_cond_tensor}

                    sample = self.ema_model(cond, lang=lang, lang_mask=lang_mask,
                                            action_cond=action_cond, verbose=False)
                    traj = sample.trajectories[0]                   # (H, a_dim + gripper + bg + K*D)
                    # Skip a[0] (pinned to current gripper pose); buffer a[1..H-1].
                    action_buffer = traj[1:, :a_dim].detach().cpu().numpy().astype(np.float32)
                    chunk_idx = 0
                    last_traj = traj
                    last_gs_np = gs_np

                a_norm = action_buffer[chunk_idx]
                action = self.dataset.normalizer.unnormalize(a_norm[None], "actions")[0]

                if os.environ.get("ECDIFF_DEBUG_ACTIONS") == "1" and last_traj is not None:
                    a0n = last_traj[0, :a_dim].detach().cpu().numpy().astype(np.float32)
                    a0 = self.dataset.normalizer.unnormalize(a0n[None], "actions")[0]
                    aH_n = last_traj[-1, :a_dim].detach().cpu().numpy().astype(np.float32)
                    aH = self.dataset.normalizer.unnormalize(aH_n[None], "actions")[0]
                    dpos_0a = float(np.linalg.norm(action[:3] - a0[:3]))
                    dpos_0H = float(np.linalg.norm(aH[:3] - a0[:3]))
                    print(f"[debug t={t:3d} chunk={chunk_idx}/{exe_steps}] "
                          f"gs.pos=({last_gs_np[0]:+.3f},{last_gs_np[1]:+.3f},{last_gs_np[2]:+.3f}) "
                          f"a0.pos=({a0[0]:+.3f},{a0[1]:+.3f},{a0[2]:+.3f}) "
                          f"a.pos=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}) "
                          f"|a-a0|={dpos_0a:.4f} |aH-a0|={dpos_0H:.4f} "
                          f"grip[a0,a,aH]=({a0[9]:+.2f},{action[9]:+.2f},{aH[9]:+.2f}) "
                          f"gs.grip={last_gs_np[9]:+.2f}",
                          flush=True)

                next_obs, reward, done, info = env.step(action)

                # Record what the model *thought* the state would look like after
                # executing this action (chunk_idx indexes into action_buffer =
                # last_traj[1:], so the corresponding predicted obs row is
                # last_traj[chunk_idx + 1]).
                if (_save_imagined or _save_imagined_recon) and last_traj is not None:
                    _pred_row = min(chunk_idx + 1, last_traj.shape[0] - 1)
                    _pred_obs_norm = last_traj[_pred_row, _obs_start:].detach().cpu().numpy()
                    _pred_obs_norm = _pred_obs_norm.reshape(K, D)
                    _pred_obs = self.dataset.normalizer.unnormalize(_pred_obs_norm, "observations")
                    # Front view = first K_per_view particles (views concatenated in order)
                    _front_pred = _pred_obs[:_K_per_view]
                    if _save_imagined:
                        _imagined_kps_front.append(_front_pred[:, :2].astype(np.float32))

                    # Full DLP decoder reconstruction of the imagined state.
                    # Decode only the front view (first K_per_view particles,
                    # first bg_per_view bg features) so the result aligns with
                    # the existing rollout RGB video.
                    if _save_imagined_recon and _dlp_model is not None:
                        try:
                            import torch as _torch
                            _dlp_dev = next(_dlp_model.parameters()).device
                            if not hasattr(_dlp_model, "_decode_sig_printed"):
                                import inspect as _inspect
                                try:
                                    _sig = _inspect.signature(_dlp_model.decode_all)
                                    print(f"[eval_rlbench] dlp.decode_all signature: {_sig}", flush=True)
                                except Exception:
                                    pass
                                _dlp_model._decode_sig_printed = True
                            # Front-view bg
                            if bg_dim > 0:
                                _pred_bg_norm = last_traj[_pred_row, _bg_start:_bg_start + bg_dim].detach().cpu().numpy()
                                _pred_bg = self.dataset.normalizer.unnormalize(
                                    _pred_bg_norm.reshape(1, -1), "bg_features"
                                )[0]
                                _front_bg = _pred_bg[:_bg_per_view]
                            else:
                                _front_bg = np.zeros((1,), dtype=np.float32)
                            # Pack front-view particles into decoder inputs.
                            # 10-D layout: [z(2), z_scale(2), z_depth(1), obj_on(1), z_features(4)]
                            _fp = _torch.from_numpy(_front_pred).float().to(_dlp_dev).unsqueeze(0)  # (1, K_per_view, D)
                            _fb = _torch.from_numpy(_front_bg).float().to(_dlp_dev).unsqueeze(0)    # (1, bg_per_view)
                            _z = _fp[:, :, 0:2]
                            _z_scale = _fp[:, :, 2:4]
                            _z_depth = _fp[:, :, 4:5]
                            # Keep obj_on as (B, K, 1) to match the encoder's
                            # packing (eval_rlbench.py:244-246 ensures trailing
                            # dim = 1 before concat into the 10D token).
                            _obj_on = _fp[:, :, 5:6]
                            _z_feat = _fp[:, :, 6:]  # (1, K_per_view, D-6)
                            # Matches the canonical decode pattern in
                            # origin/2D_DLP scripts/visualize_imagined_states.py:
                            # z_ctx=None, use 'rec' key, squeeze + clip to [0,1].
                            with _torch.no_grad():
                                _dec = _dlp_model.decode_all(
                                    _z, _z_scale, _z_feat, _obj_on, _z_depth, _fb, None,
                                    warmup=False,
                                )
                            _rec = _dec["rec"].squeeze(0).permute(1, 2, 0).cpu().numpy()
                            _rec = np.clip(_rec, 0.0, 1.0)
                            _imagined_recon_frames.append((_rec * 255).astype(np.uint8))
                        except Exception as _e:
                            if not _imagined_recon_error_printed:
                                print(f"[eval_rlbench] imagined recon decode failed: "
                                      f"{type(_e).__name__}: {_e}", flush=True)
                                _imagined_recon_error_printed = True

                # LIVE-token recon: decode the actual live-encoded post-step tokens
                # through the same pipeline. Sanity check for decoder-pipeline bugs.
                if _save_imagined_recon and _dlp_model is not None and next_obs is not None:
                    try:
                        import torch as _torch
                        _dlp_dev = next(_dlp_model.parameters()).device
                        _live_toks = np.asarray(next_obs["obs"], dtype=np.float32)
                        if _live_toks.ndim == 1:
                            _live_toks = _live_toks.reshape(K, D)
                        _live_bg_full = np.asarray(next_obs["bg_features"], dtype=np.float32)
                        _live_front = _live_toks[:_K_per_view]
                        _live_front_bg = _live_bg_full[:_bg_per_view] if bg_dim > 0 else np.zeros((1,), dtype=np.float32)

                        _fp_l = _torch.from_numpy(_live_front).float().to(_dlp_dev).unsqueeze(0)
                        _fb_l = _torch.from_numpy(_live_front_bg).float().to(_dlp_dev).unsqueeze(0)
                        _z_l = _fp_l[:, :, 0:2]
                        _z_scale_l = _fp_l[:, :, 2:4]
                        _z_depth_l = _fp_l[:, :, 4:5]
                        _obj_on_l = _fp_l[:, :, 5:6]
                        _z_feat_l = _fp_l[:, :, 6:]
                        with _torch.no_grad():
                            _dec_l = _dlp_model.decode_all(
                                _z_l, _z_scale_l, _z_feat_l, _obj_on_l, _z_depth_l, _fb_l, None,
                                warmup=False,
                            )
                        _rec_l = _dec_l["rec"].squeeze(0).permute(1, 2, 0).cpu().numpy()
                        _rec_l = np.clip(_rec_l, 0.0, 1.0)
                        _live_recon_frames.append((_rec_l * 255).astype(np.uint8))
                    except Exception as _e:
                        if not _live_recon_error_printed:
                            print(f"[eval_rlbench] live recon decode failed: "
                                  f"{type(_e).__name__}: {_e}", flush=True)
                            _live_recon_error_printed = True

                last_reward = float(reward)
                last_error = info.get("error")
                if last_error:
                    # Print the offending action so we can diagnose InvalidActionError /
                    # IKError / ConfigurationPathError. action = [pos(3), rot6d(6), grip(1)].
                    pos = action[:3]
                    print(f"[eval_rlbench]   t={t} {last_error}: "
                          f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                          f"rot6d_norms=({np.linalg.norm(action[3:6]):.3f},{np.linalg.norm(action[6:9]):.3f}) "
                          f"grip={action[9]:+.3f}",
                          flush=True)
                if next_obs is not None:
                    obs_dict = next_obs
                chunk_idx += 1
                t += 1

            success = 1.0 if last_reward >= 0.5 else 0.0
            successes.append(success)
            lengths.append(t)
            vnum = obs_dict.get("variation_number")
            if vnum is not None:
                per_variation.setdefault(int(vnum), []).append(success)
            end_reason = (
                "success" if success > 0
                else f"error:{last_error}" if last_error
                else ("timeout" if t >= max_steps else "task_terminal")
            )
            print(f"[eval_rlbench] ep={ep} steps={t} success={int(success)} "
                  f"end={end_reason} lang={obs_dict.get('language')!r}", flush=True)

            # Dump recorded front-cam frames (plain + kp overlay) to mp4
            try:
                import imageio.v2 as _imageio
                frames, front_toks = env.pop_recorded_frames()
                if frames:
                    video_dir = os.path.join(self.logdir, "eval_videos", f"step_{self.step}")
                    os.makedirs(video_dir, exist_ok=True)
                    tag = "success" if success > 0 else "fail"

                    # Plain front view.
                    plain_path = os.path.join(video_dir, f"ep{ep:02d}_{tag}.mp4")
                    _imageio.mimsave(plain_path, frames, fps=20, macro_block_size=1)
                    print(f"[eval_rlbench] saved video: {plain_path}", flush=True)

                    # KP overlay: draw the front-view particle z-positions per frame.
                    if front_toks and len(front_toks) == len(frames):
                        try:
                            from utils.util_func import plot_keypoints_on_image
                            import torch as _torch
                            kp_frames = []
                            for f, tok in zip(frames, front_toks):
                                kp_xy = _torch.from_numpy(tok[:, :2])   # (K_front, 2) in [-1,1]
                                img_chw = _torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                                kp_img = plot_keypoints_on_image(
                                    kp_xy, img_chw,
                                    radius=2, thickness=1,
                                    kp_range=(-1, 1), plot_numbers=False,
                                )
                                kp_frames.append(np.asarray(kp_img, dtype=np.uint8))
                            kp_path = os.path.join(video_dir, f"ep{ep:02d}_{tag}_kp.mp4")
                            _imageio.mimsave(kp_path, kp_frames, fps=20, macro_block_size=1)
                            print(f"[eval_rlbench] saved video: {kp_path}", flush=True)
                        except Exception as e:
                            print(f"[eval_rlbench] kp overlay skipped: {type(e).__name__}: {e}",
                                  flush=True)

                    # Imagined-state DLP reconstruction video: each frame is
                    # the front-view RGB the DLP decoder produces from the
                    # model's predicted particle tokens for that step.
                    if _save_imagined_recon and _imagined_recon_frames:
                        try:
                            recon_path = os.path.join(video_dir, f"ep{ep:02d}_{tag}_imagined_recon.mp4")
                            _imageio.mimsave(recon_path, _imagined_recon_frames, fps=20, macro_block_size=1)
                            print(f"[eval_rlbench] saved imagined recon video: {recon_path}", flush=True)
                        except Exception as e:
                            print(f"[eval_rlbench] imagined recon save failed: {type(e).__name__}: {e}", flush=True)

                    # LIVE-token recon: DLP decoder rendering of the actual live
                    # particle tokens at each step. Should look like the real
                    # scene if the decoder pipeline is correct.
                    if _save_imagined_recon and _live_recon_frames:
                        try:
                            live_path = os.path.join(video_dir, f"ep{ep:02d}_{tag}_live_recon.mp4")
                            _imageio.mimsave(live_path, _live_recon_frames, fps=20, macro_block_size=1)
                            print(f"[eval_rlbench] saved live recon video: {live_path}", flush=True)
                        except Exception as e:
                            print(f"[eval_rlbench] live recon save failed: {type(e).__name__}: {e}", flush=True)

                    # Imagined-state KP overlay: each frame shows where the model
                    # predicted the front-view particles would be at that step,
                    # drawn on the actual post-step RGB. Divergence from the real
                    # KP overlay => the model's scene dynamics imagination is
                    # wrong (even if its action reconstruction on training data
                    # is accurate).
                    if _save_imagined and _imagined_kps_front:
                        try:
                            from utils.util_func import plot_keypoints_on_image
                            import torch as _torch
                            im_frames = []
                            # frames[0] is the pre-step frame; imagined_kps[i]
                            # corresponds to frames[i+1] (the post-step frame).
                            n_pairs = min(len(_imagined_kps_front), len(frames) - 1)
                            for i in range(n_pairs):
                                kp_xy = _torch.from_numpy(_imagined_kps_front[i])
                                img_chw = _torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float() / 255.0
                                kp_img = plot_keypoints_on_image(
                                    kp_xy, img_chw,
                                    radius=2, thickness=1,
                                    kp_range=(-1, 1), plot_numbers=False,
                                )
                                im_frames.append(np.asarray(kp_img, dtype=np.uint8))
                            if im_frames:
                                im_path = os.path.join(video_dir, f"ep{ep:02d}_{tag}_imagined_kp.mp4")
                                _imageio.mimsave(im_path, im_frames, fps=20, macro_block_size=1)
                                print(f"[eval_rlbench] saved imagined KP video: {im_path}", flush=True)
                        except Exception as e:
                            print(f"[eval_rlbench] imagined KP save failed: {type(e).__name__}: {e}", flush=True)
            except Exception as e:
                print(f"[eval_rlbench] video save failed: {type(e).__name__}: {e}", flush=True)

            # GT replay: reset to same demo initial state, execute the recorded
            # demo actions open-loop, save alongside the rollout for side-by-side
            # comparison. Pop happens after the rollout save above, so the
            # buffer is already empty here.
            if _save_gt and _demo is not None and _pkl_actions_all is not None:
                try:
                    env.reset(demo=_demo)
                    _demo_actions = np.asarray(_pkl_actions_all[_demo_idx])  # (T, a_dim)
                    _demo_steps = 0
                    for _dt in range(_demo_actions.shape[0]):
                        _a_demo = _demo_actions[_dt]
                        if np.allclose(_a_demo, 0.0):
                            break
                        try:
                            _, _r_demo, _done_demo, _info_demo = env.step(_a_demo)
                            _demo_steps += 1
                            if _done_demo:
                                break
                        except Exception:
                            continue
                    import imageio.v2 as _imageio
                    demo_frames, _ = env.pop_recorded_frames()
                    if demo_frames:
                        video_dir = os.path.join(self.logdir, "eval_videos", f"step_{self.step}")
                        os.makedirs(video_dir, exist_ok=True)
                        _demo_var = int(getattr(_demo, "variation_number", -1))
                        demo_path = os.path.join(
                            video_dir,
                            f"ep{ep:02d}_demo_ep{_demo_idx}_var{_demo_var}.mp4",
                        )
                        _imageio.mimsave(demo_path, demo_frames, fps=20, macro_block_size=1)
                        print(f"[eval_rlbench] saved GT demo video: {demo_path} "
                              f"(replayed {_demo_steps} actions)", flush=True)
                except Exception as e:
                    print(f"[eval_rlbench] GT demo replay failed: "
                          f"{type(e).__name__}: {e}", flush=True)

        env.shutdown()

        out = {
            "sim/success_rate": float(np.mean(successes)) if successes else 0.0,
            "sim/avg_len": float(np.mean(lengths)) if lengths else 0.0,
            "sim/n_episodes": int(len(successes)),
        }
        for vnum, vals in per_variation.items():
            out[f"sim/var{vnum}/success_rate"] = float(np.mean(vals))
        return out



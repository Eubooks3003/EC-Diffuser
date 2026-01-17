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
from dlp_utils import log_rgb_voxels

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
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10, front_bg=None, side_bg=None):
        '''
            renders training points
        '''

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
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        savepath = os.path.join(self.logdir, f'_sample-reference.ply')
        self.renderer.composite(savepath, observations, front_bg=front_bg, side_bg=side_bg)

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

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations, front_bg=front_bg, side_bg=side_bg)
    

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

        save_videos=True,
        video_dir=None,
        video_fps=20,
        video_cams=("agentview",),

        renderer_3d=None,
        render_debug=True,
        render_debug_steps=(0,),
        exe_steps=1,  # NEW: number of actions to execute from each plan before replanning
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
        print(f"[eval_mimicgen_rollouts] ACTION CHUNKING: exe_steps={exe_steps}, horizon={self.dataset.horizon}")
        if exe_steps > 1:
            print(f"[eval_mimicgen_rollouts] Will execute {exe_steps} actions per plan before replanning")
        else:
            print(f"[eval_mimicgen_rollouts] WARNING: exe_steps=1 means replanning every step (no chunking)")
        print("=" * 60, flush=True)

        for ep in range(n_episodes):
            print(f"\n[eval] Episode {ep+1}/{n_episodes}")
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
            )

            obs_vec = envw.reset()

            if render_debug and renderer_3d is not None:
                # _render_tokens_debug(tag=f"ep_{ep:02d}/reset_obs", obs_vec_flat=obs_vec, horizon_step=0)

                vox = envw.last_vox  # np [C,D,H,W]
                print("GT VOX: ", vox.shape )
                    # avg_rgb
                log_rgb_voxels(
                    name=f"eval_debug/ep_{ep:02d}/gt_vox_rgb",
                    rgb_vol=vox,          # [3,D,H,W]
                    alpha_vol=None,
                    KPx=None,
                    step=int(200),
                    mode="splat",
                    topk=60000,
                    alpha_thresh=0.05,
                    pad=2.0,
                    show_axes=True,
                )
                # if you want to also see the goal:
                # if getattr(envw, "goal_vec", None) is not None:
                #     _render_tokens_debug(tag=f"ep_{ep:02d}/reset_goal", obs_vec_flat=envw.goal_vec, horizon_step=0)

            ep_ret = 0.0
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

            t = 0
            while t < max_steps:
                # Check if we need to replan (no buffer, or exhausted exe_steps actions)
                need_replan = (action_buffer is None) or (action_idx >= exe_steps) or (action_idx >= action_buffer.shape[0])

                if need_replan:
                    # ====== PLANNING PHASE ======
                    cond_np = envw.make_cond(obs_vec, horizon=self.dataset.horizon)
                    cond = {k: norm_obs(v)[None, :] for k, v in cond_np.items()}

                    # === GOAL CONDITIONING DIAGNOSTIC (first timestep only) ===
                    if t == 0:
                        obs_cond = cond_np[0]
                        goal_cond = cond_np[self.dataset.horizon - 1]
                        diff = np.abs(goal_cond - obs_cond)
                        print(f"\n[GOAL DIAGNOSTIC ep={ep}]")
                        print(f"  obs vs goal diff: mean={diff.mean():.4f}, max={diff.max():.4f}, std={diff.std():.4f}")
                        print(f"  obs range: [{obs_cond.min():.4f}, {obs_cond.max():.4f}]")
                        print(f"  goal range: [{goal_cond.min():.4f}, {goal_cond.max():.4f}]")

                        cond_no_goal = {0: cond[0], self.dataset.horizon - 1: cond[0].clone()}
                        sample_no_goal = self.ema_model(cond_no_goal, verbose=False)
                        a0_no_goal = sample_no_goal.trajectories[0, 0, :a_dim].detach().cpu().numpy()
                        sample_with_goal = self.ema_model(cond, verbose=False)
                        a0_with_goal = sample_with_goal.trajectories[0, 0, :a_dim].detach().cpu().numpy()
                        action_diff = np.abs(a0_with_goal - a0_no_goal)
                        print(f"  Action WITH goal:    [{', '.join([f'{x:.4f}' for x in a0_with_goal])}]")
                        print(f"  Action WITHOUT goal: [{', '.join([f'{x:.4f}' for x in a0_no_goal])}]")
                        print(f"  Action diff: mean={action_diff.mean():.4f}, max={action_diff.max():.4f}")
                        if action_diff.mean() < 0.05:
                            print(f"  WARNING: Actions barely change with/without goal!")
                        print()

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
                if done:
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
    

